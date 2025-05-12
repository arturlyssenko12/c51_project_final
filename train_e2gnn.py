import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from ase.io import read
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN.E2GNN import E2GNN
from graph_constructor import AtomsToGraphs

from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric

from loaders.loaders import create_atomic_data_list, create_e2gnn_data_list, move_batch,collate_fn_e2gnn

# ------------------- Config -------------------

TEACHER_PATH = "2023-12-10-mace-128-L0_energy_epoch-249.model"
XYZ_PATH_TRAIN = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
XYZ_PATH_TEST = "BOTNet-datasets/dataset_3BPA/test_300K.xyz"
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
FORCE_WEIGHT = 100.0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cuda:1"

target_dat = torch.load("/home/alyssenko/c51_project/utils/all_hessians.pt")



# ------------------- Load Models -------------------

teacher = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False
teacher_dtype = next(teacher.parameters()).dtype

student = E2GNN(
    hidden_channels=128,
    num_layers=3,
    num_rbf=32,
    cutoff=6.0,
    max_neighbors=20,
    use_pbc=False,
    otf_graph=True
).to(device=device, dtype=torch.float).train()


# ------------------- Optimizer -------------------

optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=2e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-6)
loss_fn_energy = nn.MSELoss()
loss_fn_force = nn.MSELoss()

# ------------------- Load Data -------------------

atoms_list = read(XYZ_PATH_TRAIN, ":")
atoms_list_val = read(XYZ_PATH_TEST, ":")

teacher_data = create_atomic_data_list(atoms_list, teacher, "teacher")
student_data = create_e2gnn_data_list(atoms_list, target_dat, "student",device)



teacher_loader = torch_geometric.dataloader.DataLoader(
    teacher_data, batch_size=BATCH_SIZE, shuffle=True
)
student_loader = DataLoader(
    student_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_e2gnn
)

min_len = min(len(teacher_loader), len(student_loader))

# ------------------- Training -------------------

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0

    loop = tqdm(
        zip(teacher_loader, student_loader),
        total=min_len,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for i, (batch_teacher, batch_student) in enumerate(loop):
        if i >= min_len:
            break

        batch_teacher = move_batch(batch_teacher, device, dtype=teacher_dtype)
        batch_student = move_batch(batch_student, device, dtype=torch.float)

        for b in [batch_teacher, batch_student]:
            if b.edge_index is not None:
                b.edge_index = b.edge_index.to(torch.long)
            if b.batch is not None:
                b.batch = b.batch.to(torch.long)
            if hasattr(b, "head") and b.head is not None:
                b.head = b.head.to(torch.long)

        batch_teacher.positions.requires_grad_(True)
        teacher_out = teacher(batch_teacher.to_dict())
        target_energy = teacher_out["energy"].detach().to(dtype=torch.float)
        target_forces = teacher_out["forces"].detach().to(dtype=torch.float)

        predicted_energy, predicted_forces = student(batch_student)
        energy_loss = loss_fn_energy(predicted_energy, target_energy)   
        force_loss = loss_fn_force(predicted_forces, target_forces)
        total_batch_loss = energy_loss + FORCE_WEIGHT * force_loss
        # if epoch ==10:
        #     import pdb
        #     pdb.set_trace()
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()

        loop.set_postfix({
            "Loss": total_batch_loss.item(),
            "E": energy_loss.item(),
            "F": force_loss.item()
        })

    scheduler.step(total_loss / min_len)
    print(f"Epoch {epoch+1}: Loss={total_loss/min_len:.4f}, Energy={total_energy_loss/min_len:.4f}, Force={total_force_loss/min_len:.4f}")

# ------------------- Save -------------------

torch.save(student, "fine_tuned_student_in_memory.model")
print("\u2705 Training complete. Model saved to fine_tuned_student_in_memory.model")