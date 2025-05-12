import os
import sys
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from ase.io import read
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN.E2GNN import E2GNN
from loaders.loaders import AtomsDataset, collate_fn_e2gnn, move_batch

sweep_configuration = {
    "name": "e2gnn_force_distill-sweep",
    "method": "bayes",
    "metric": {"name": "val_loss_total", "goal": "minimize"},
    "parameters": {
        "LEARNING_RATE": {"values": [1e-4, 5e-4, 1e-3, 2e-3]},
        "ENERGY_WEIGHT": {"values": [0, 5]},
        "FORCE_WEIGHT": {"values": [50.0, 100.0]},
        "GRAD_CLIP_NORM": {"values": [5.0, 10.0]},
        "EMA_DECAY": {"values": [0.99, 0.995, 0.999]},
        "WEIGHT_DECAY": {"values": [1e-5, 2e-6, 1e-6]},
        "LR_SCHEDULER_PATIENCE": {"values": [5, 10]},
        "LR_SCHEDULER_FACTOR": {"values": [0.5, 0.8]},
        "LR_MIN": {"values": [1e-6, 5e-7]},
        "BATCH_SIZE": {"values": [4,8,16,32]},
        "EPOCHS": {"values": [100,200]},
    }
}

# sweep_configuration = {
#     "method": "bayes",
#     "metric": {"name": "val_loss_total", "goal": "minimize"},
#     "parameters": {
#         "LR_SCHEDULER_PATIENCE": {
#             "min": 3,
#             "max": 20,
#             "distribution": "int_uniform"
#         },
#         "LR_SCHEDULER_FACTOR": {
#             "min": 0.25,
#             "max": 0.8,
#             "distribution": "uniform"
#         },
#         "GRAD_CLIP_NORM": {
#             "min": 3,
#             "max": 10,
#             "distribution": "int_uniform"
#         },
#         "ENERGY_WEIGHT": {"values": [0, 5]},
#         "LEARNING_RATE": {
#             "min": 0.00005,
#             "max": 0.01,
#             "distribution": "uniform"
#         },
#         "WEIGHT_DECAY": {
#             "min": 5e-7,
#             "max": 0.00002,
#             "distribution": "uniform"
#         },
#         "FORCE_WEIGHT": {
#             "min": 25,
#             "max": 100,
#             "distribution": "int_uniform"
#         },
#         "BATCH_SIZE": {"values": [4,8,16,32]},
#         "EMA_DECAY": {
#             "min": 0.495,
#             "max": 0.999,
#             "distribution": "uniform"
#         },
#         "LR_MIN": {
#             "min": 5e-7,
#             "max": 0.000002,
#             "distribution": "uniform"
#         },
#         "EPOCHS": {
#             "min": 50,
#             "max": 200,
#             "distribution": "int_uniform"
#         }
#     }
# }


XYZ_PATH = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
XYZ_PATH_TEST = "BOTNet-datasets/dataset_3BPA/test_300K.xyz"
TARGET_PATH = "/home/alyssenko/c51_project/utils/all_hessians.pt"

loss_fn_energy = nn.MSELoss()
loss_fn_force = nn.MSELoss()


def evaluate(loader, model, device):
    model.eval()
    e_true, e_pred, f_true, f_pred, h_true, h_pred = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = move_batch(batch, device, torch.float)
        e_p, f_p = model(batch)
        e_true.append(batch.y.cpu())
        e_pred.append(e_p.cpu())
        f_true.append(batch.force.cpu())
        f_pred.append(f_p.cpu())



    e_loss = ((torch.cat(e_true).squeeze() - torch.cat(e_pred).squeeze())**2).mean().item()
    f_loss = ((torch.cat(f_true).view(-1, 3) - torch.cat(f_pred).view(-1, 3))**2).mean().item()
    return e_loss, f_loss

def main():
    wandb.init(project="e2gnn_force_distill-sweep")
    config = wandb.config
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:1"

    student = E2GNN(
        hidden_channels=128,
        num_layers=3,
        num_rbf=32,
        cutoff=4.5,
        max_neighbors=15,
        use_pbc=False,
        otf_graph=True,
        num_elements=9
    ).to(device).to(dtype=torch.float).train()

    ema_student = AveragedModel(student, avg_fn=lambda avg, cur, n: config.EMA_DECAY * avg + (1 - config.EMA_DECAY) * cur)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
                                                           patience=config.LR_SCHEDULER_PATIENCE, min_lr=config.LR_MIN)

    atoms_list = read(XYZ_PATH, ":")
    target_dict = torch.load(TARGET_PATH)
    train_dataset = AtomsDataset(atoms_list, target_dict, device)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_e2gnn)

    atoms_test_full = read(XYZ_PATH_TEST, ":")
    target_len = min(len(target_dict["energy"]), len(target_dict["forces"]))
    atoms_test_full = atoms_test_full[:target_len]
    atoms_val, _ = train_test_split(atoms_test_full, test_size=0.5, random_state=42)
    target_val = {
        "energy": target_dict["energy"][:len(atoms_val)],
        "forces": target_dict["forces"][:len(atoms_val)],
        "hessian": target_dict["hessian"][:len(atoms_val)],
    }
    val_loader = DataLoader(AtomsDataset(atoms_val, target_val, device), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_e2gnn)

    for epoch in range(config.EPOCHS):
        student.train()
        total_loss = total_energy_loss = total_force_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            batch = move_batch(batch, device, torch.float)
            pred_energy, pred_forces = student(batch)
            energy_loss = loss_fn_energy(pred_energy, batch.y)
            force_loss = loss_fn_force(pred_forces, batch.force)

            
            
            loss = config.ENERGY_WEIGHT * energy_loss + config.FORCE_WEIGHT * force_loss 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            ema_student.update_parameters(student)
            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()

        scheduler.step(total_loss / len(train_loader))
        val_e, val_f = evaluate(val_loader, ema_student.module, device)
        wandb.log({
            "val_loss_energy": val_e,
            "val_loss_force": val_f,
            "val_loss_total": val_e * config.ENERGY_WEIGHT + val_f * config.FORCE_WEIGHT,
            "train_loss_energy": total_energy_loss / len(train_loader),
            "train_loss_force": total_force_loss / len(train_loader),
            "train_loss_total": total_loss / len(train_loader)
        })

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="e2gnn_force_distill-sweep")
    wandb.agent(sweep_id, function=main)
