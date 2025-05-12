import os
import sys
import random
import multiprocessing as mp
# combine imports for read
from ase.io import read
import torch
from torch import compile as torch_compile
from torch.func import vjp, vmap
from torch.nn import MSELoss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn.pool import radius_graph
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from loaders.loaders import AtomsToGraphs, collate_fn_e2gnn, move_batch, AtomsDataset
# Add E2GNN source path
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN

# Ensure 'spawn' start method for safe CUDA in DataLoader workers
mp.set_start_method('spawn', force=True)

device = "cuda:0"
torch.set_default_dtype(torch.float)

# --- Hyperparameters --------------------------------------
NUM_EPOCHS, BATCH_SIZE = 100, 64
LR, ENERGY_W, FORCE_W, HESSIAN_W = 1e-3, 5.0, 100.0, 400.0
GRAD_NORM, EMA_DECAY, WEIGHT_DECAY = 10.0, 0.999, 2e-6
PATIENCE, FACTOR, LR_MIN, HESS_ROWS = 5, 0.8, 1e-6, 128

# Loss functions
loss_e = MSELoss(reduction='sum')
loss_f = MSELoss(reduction='sum')
loss_h = MSELoss(reduction='sum')

# --- Helper Functions ------------------------------------
def combine_xyz_files(paths):
    atoms = []
    for p in paths:
        atoms.extend(read(p, ":"))
    return atoms

def plot_corr(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.3, s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def combine_target_dicts(paths):
    combined = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        for k in combined:
            combined[k].extend(d[k])
    for k in combined:
        combined[k] = [x.cpu() for x in combined[k]]
    return combined


def compute_sampled_hessian(batch, mdl, loss_fn, max_samples=HESS_ROWS):
    # unpack a single example
    data   = batch.to_data_list()[0]
    single = Batch.from_data_list([data])
    single = move_batch(single, device, torch.float)

    # make positions differentiable
    single.pos = single.pos.detach().clone().requires_grad_(True)

    # forward pass → forces
    _, forces = mdl(single)
    flat_f    = forces.view(-1)

    # pre-compute ground truth Hessian as (n_dof, n_dof)
    n         = flat_f.numel()
    H_gt      = single.hessian.view(n, n)

    # sample a few rows
    idx       = torch.randperm(n, device=flat_f.device)[:max_samples]
    losses    = []

    for i in idx:
        # set up one-hot to pick out f_i
        go     = torch.zeros_like(flat_f)
        go[i]  = 1.0

        # ∂f_i/∂x → this is –H_row_i
        g      = torch.autograd.grad(
                   flat_f, single.pos,
                   grad_outputs=go,
                   retain_graph=True,
                   create_graph=True
                 )[0]

        # flatten & flip sign so that +g_flat = +H_row
        g_flat = g.view(-1)
        pred   = -g_flat       # because ∇F = –H
        true   = H_gt[i]       # full row: shape (n,)

        # vector-to-vector loss
        losses.append(loss_fn(pred, true))

    return sum(losses) / len(losses) if losses else torch.tensor(0., device=flat_f.device)



def evaluate(loader, mdl):
    mdl.eval()
    se = sf = sh = 0.0
    with torch.no_grad():
        for batch in loader:
            b = move_batch(batch, device, torch.float)
            e_p, f_p = mdl(b)
            se += loss_e(e_p, b.y).item()
            sf += loss_f(f_p, b.force).item()
    for batch in loader:
        sh += compute_sampled_hessian(batch, mdl, loss_h).item()
    return se, sf, sh


def evaluate_full(loader, mdl, device, hess_rows=HESS_ROWS):
    mdl.eval()
    e_true = []; e_pred = []
    f_true = []; f_pred = []
    h_true = []; h_pred = []
    for batch in loader:
        for data in batch.to_data_list():
            single = Batch.from_data_list([data])
            single = move_batch(single, device, torch.float)
            single.pos = single.pos.detach().clone().requires_grad_(True)
            e_out, f_out = mdl(single)
            e_true.append(single.y.item())
            e_pred.append(e_out.item())
            f_flat = f_out.view(-1)
            f_true.extend(single.force.view(-1).cpu().tolist())
            f_pred.extend(f_flat.cpu().tolist())
            h_flat = single.hessian.view(-1)
            idx = torch.randperm(f_flat.numel(), device=device)[:hess_rows]
            for i in idx:
                go = torch.zeros_like(f_flat); go[i]=1.0
                g = torch.autograd.grad(f_flat, single.pos, grad_outputs=go,
                                        retain_graph=True)[0].view(-1)
                h_pred.append(g[i].item())
                h_true.append(h_flat[i].item())
    return e_true, e_pred, f_true, f_pred, h_true, h_pred



def main():

    # Data loading
    XYZ_TRAIN = [
        "BOTNet-datasets/dataset_3BPA/train_300K.xyz",
        "BOTNet-datasets/dataset_3BPA/train_mixedT.xyz",
        "BOTNet-datasets/dataset_3BPA/test_dih.xyz",
    ]

    PT_TRAIN = [
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_300K.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_mixedT.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_dih.pt",
    ]

    XYZ_TEST = [
        "BOTNet-datasets/dataset_3BPA/test_300K.xyz",
    ]

    PT_TEST = [
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_300K.pt",
    ]


    # Build train set + compute Hessian std
    train_atoms = combine_xyz_files(XYZ_TRAIN)
    train_tgt   = combine_target_dicts(PT_TRAIN)
    # call script in scratch.ipynb to inspect the data what this dict is looking like


    all_h = torch.cat([h.flatten() for h in train_tgt['hessian']])
    hessian_std = all_h.std()

    train_ds = AtomsDataset(train_atoms, train_tgt, device, hessian_scale=hessian_std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_e2gnn)#,num_workers=8) 
    mean_e, std_e, h_scale = train_ds.energy_mean, train_ds.energy_std, train_ds.hessian_scale

    test_atoms = combine_xyz_files(XYZ_TEST)
    test_tgt   = combine_target_dicts(PT_TEST)

    # 2) split 50/50 by index
    indices      = list(range(len(test_atoms)))
    idx_val, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    # 3) slice into val vs. test
    val_atoms  = [test_atoms[i] for i in idx_val]
    test_atoms = [test_atoms[i] for i in idx_test]

    val_tgt = { k: [test_tgt[k][i] for i in idx_val]   for k in test_tgt }
    test_tgt= { k: [test_tgt[k][i] for i in idx_test]  for k in test_tgt }

    # 4) build datasets & loaders
    val_ds    = AtomsDataset(val_atoms,  val_tgt,  device,
                            energy_mean=mean_e, energy_std=std_e, hessian_scale=h_scale)
    test_ds   = AtomsDataset(test_atoms, test_tgt, device,
                            energy_mean=mean_e, energy_std=std_e, hessian_scale=h_scale)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn_e2gnn)# ,num_workers=8) 
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn_e2gnn)#,num_workers=8) 


    # Model
    model = E2GNN(
        hidden_channels=128,
        num_layers=3,
        num_rbf=32,
        cutoff=4.5,
        max_neighbors=15,
        use_pbc=False,
        otf_graph=True,
        num_elements=9
    ).to(device)

    ema_model = AveragedModel(model,
        avg_fn=lambda a, c, n: EMA_DECAY * a + (1-EMA_DECAY) * c
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=FACTOR, patience=PATIENCE, min_lr=LR_MIN)

    # Training loop
    train_losses = {"energy":[],"force":[],"hessian":[],"total":[]}
    val_losses   = {"energy":[],"force":[],"hessian":[]}
    for epoch in range(NUM_EPOCHS):
        model.train()
        tot_e=tot_f=tot_h=tot_all=0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            batch = move_batch(batch, device, torch.float)
            e_p, f_p = model(batch)
            e_l = loss_e(e_p, batch.y)
            f_l = loss_f(f_p, batch.force)
            h_l = compute_sampled_hessian(batch, model, loss_h)
            # if epoch < 5:
            #     HESSIAN_W = 0
            # elif epoch < 20:
            #     HESSIAN_W = 10
            # else:
            #     HESSIAN_W = 200
            loss = ENERGY_W*e_l + FORCE_W*f_l + HESSIAN_W*h_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            opt.step()
            # if epoch>=NUM_EPOCHS//4: 
            ema_model.update_parameters(model)
            tot_e+=e_l.item(); tot_f+=f_l.item(); tot_h+=h_l.item(); tot_all+=loss.item()
        n = len(train_loader.dataset)
        train_losses['energy'].append(tot_e/n)
        train_losses['force'].append(tot_f/n)
        train_losses['hessian'].append(tot_h/n)
        train_losses['total'].append(tot_all/n)
        # sched.step(tot_all/n)
        print(f"Train mean E/F/H/Total: {tot_e/n:.4f}/{tot_f/n:.4f}/{tot_h/n:.4f}/{tot_all/n:.4f}")
        se, sf, sh = evaluate(val_loader, model)
        nval = len(val_loader.dataset)
        # sched.step(sf / nval)
        sched.step((se+ sf+ sh) / nval)
        val_losses['energy'].append(se/nval)
        val_losses['force'].append(sf/nval)
        val_losses['hessian'].append(sh/nval)
        print(f"Val mean E/F/H: {se/nval:.4f}/{sf/nval:.4f}/{sh/nval:.4f}")
        print(f"Val Force MAE: {sf/nval:.4f}")
    torch.save(ema_model.module, "e2gnn_student_supervised_HESSIAN.model")
    # print("Model saved")
    # plot
    for comp in ["energy","force","hessian","total"]:
        plt.figure(figsize=(8,5))
        plt.plot(train_losses[comp],label="Train")
        if comp in val_losses: plt.plot(val_losses[comp],label="Val")
        plt.xlabel("Epoch"); plt.ylabel(f"{comp.capitalize()} Loss"); plt.yscale("log"); plt.legend(); plt.tight_layout()
        plt.savefig(f"loss_{comp}.png"); plt.close()
    # test_e_true, test_e_pred, \
    # test_f_true, test_f_pred, \
    # test_h_true, test_h_pred = evaluate_full(
    #     test_loader,
    #     ema_model.module,
    #     device
    # )
    # # test_e_true, test_e_pred, test_f_true, test_f_pred, test_h_true, test_h_pred = evaluate_full(test_loader, student)
    # plot_corr(test_e_true, test_e_pred, "Energy Correlation", "True Energy", "Predicted Energy", "correlation_plot_energy.png")
    # plot_corr(test_f_true, test_f_pred, "Force Correlation", "True Force", "Predicted Force", "correlation_plot_force.png")
    # plot_corr(test_h_true, test_h_pred, "Hessian Correlation", "True Hessian", "Predicted Hessian", "correlation_plot_hessian.png")
    # print("Correlation plots saved")

if __name__ == "__main__":
    main()