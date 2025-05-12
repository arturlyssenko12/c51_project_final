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
import numpy as np
from loaders.loaders import AtomsToGraphs, collate_fn_e2gnn, move_batch, AtomsDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Add E2GNN source path
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN


import time
import pandas as pd

def evaluate_full(loader, mdl, device, dt_fs, csv_path, hess_rows=256):
    """
    Runs through the loader, computes predictions, and logs
    throughput in ns/day based on dt_fs (timestep in fs).
    
    Args:
        loader: DataLoader yielding batches of graphs
        mdl:       the compiled/EMA model
        device:    e.g. "cuda:0"
        dt_fs:     timestep in femtoseconds (e.g. 1.0 for 1 fs)
        csv_path:  path to write CSV with column 'ns_day'
        hess_rows: how many Hessian rows to sample per graph
    Returns:
        (e_true, e_pred, f_true, f_pred, h_true, h_pred)
    """
    # mdl.eval()
    e_true, e_pred = [], []
    f_true, f_pred = [], []
    # h_true, h_pred = [], []

    n_steps = 0
    t0 = time.time()

    for batch in tqdm(loader, desc="Evaluating"):
        for data in batch.to_data_list():
            n_steps += 1
            single = Batch.from_data_list([data])
            single = move_batch(single, device, torch.float)
            single.pos = single.pos.detach().clone().requires_grad_(True)

            # energy & force
            e_out, f_out = mdl(single)
            e_true.append(single.y.item())
            e_pred.append(e_out.item())

            f_flat = f_out.view(-1)
            f_true.extend(single.force.view(-1).cpu().tolist())
            f_pred.extend(f_flat.cpu().tolist())

            # # ground truth Hessian
            # n_dof = f_flat.numel()
            # H_gt = single.hessian.view(n_dof, n_dof)
            # idx = torch.randperm(n_dof, device=device)[:hess_rows]

            # for i in idx:
            #     go = torch.zeros_like(f_flat); go[i] = 1.0
            #     g = torch.autograd.grad(
            #         f_flat, single.pos,
            #         grad_outputs=go,
            #         retain_graph=True
            #     )[0].view(-1)
            #     pred_row = -g
            #     true_row = H_gt[i]
            #     h_pred.extend(pred_row.cpu().tolist())
            #     h_true.extend(true_row.cpu().tolist())

    wall_seconds = time.time() - t0

    # compute simulated time in nanoseconds
    sim_ns = n_steps * dt_fs * 1e-6  # fs → ns
    # compute throughput in ns/day
    ns_per_day = sim_ns / wall_seconds * 86400.0

    # log to CSV
    df = pd.DataFrame([{'ns_day': ns_per_day}])
    df.to_csv(csv_path, index=False)

    print(f"Processed {n_steps} steps in {wall_seconds:.1f}s → {ns_per_day:.2f} ns/day")

    return e_true, e_pred, f_true, f_pred#, h_true, h_pred

def combine_xyz_files(paths):
    atoms = []
    for p in paths:
        atoms.extend(read(p, ":"))
    return atoms

def plot_corr(x, y, title, xlabel, ylabel, filename=None):
    # convert to numpy
    x = np.asarray(x)
    y = np.asarray(y)

    # compute identical limits with 5% padding
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    margin = (vmax - vmin) * 0.05
    lims = (vmin - margin, vmax + margin)

    # compute metrics
    mae = mean_absolute_error(y, x)
    mse = mean_squared_error(y, x)

    # start plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.3, s=5)
    # perfect correlation line
    plt.plot(lims, lims, linestyle='--', linewidth=1, color='black')

    # axes, labels, title
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # metrics textbox
    textstr = f"MAE = {mae:.3e}\nMSE = {mse:.3e}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.gca().text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=props
    )

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def combine_target_dicts(paths):
    combined = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        for k in combined:
            combined[k].extend(d[k])
    for k in combined:
        combined[k] = [x.cpu() for x in combined[k]]
    return combined


device = "cuda:0"

# load and put on device
ema_model = torch.load("e2gnn_student_supervised_HESSIAN.model", map_location=device)
ema_model.to(device)

# switch to eval‐mode
ema_model.eval()

XYZ_TEST = [
    "/home/alyssenko/c51_project/BOTNet-datasets/dataset_3BPA/test_600K.xyz",
]

PT_TEST = [
    "/home/alyssenko/c51_project/BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_600K.pt",
]

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


train_atoms = combine_xyz_files(XYZ_TRAIN)
train_tgt   = combine_target_dicts(PT_TRAIN)
all_h = torch.cat([h.flatten() for h in train_tgt['hessian']])
hessian_std = all_h.std()
train_ds = AtomsDataset(train_atoms, train_tgt, device, hessian_scale=hessian_std,plot_hist=False)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_fn_e2gnn)#,num_workers=8) 
mean_e, std_e, h_scale = train_ds.energy_mean, train_ds.energy_std, train_ds.hessian_scale

test_atoms = combine_xyz_files(XYZ_TEST)
test_tgt   = combine_target_dicts(PT_TEST)
test_ds   = AtomsDataset(test_atoms, test_tgt, device,
                            energy_mean=mean_e, energy_std=std_e, hessian_scale=h_scale)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_e2gnn)


test_e_true, test_e_pred, \
test_f_true, test_f_pred= evaluate_full(
    test_loader,
    ema_model,
    device,
    1,
    "results_for_analysis/e2gnn_hessian/e2gnn_hessian_inference.csv")

plot_corr(test_e_true, test_e_pred, "Energy Correlation", "True Energy", "Predicted Energy", "results_for_analysis/e2gnn_hessian/correlation_plot_energy.png")
plot_corr(test_f_true, test_f_pred, "Force Correlation", "True Force", "Predicted Force", "results_for_analysis/e2gnn_hessian/correlation_plot_force.png")