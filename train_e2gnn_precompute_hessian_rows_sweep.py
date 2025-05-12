#!/usr/bin/env python3
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from ase.io import read

from loaders.loaders import AtomsToGraphs, collate_fn_e2gnn, move_batch, AtomsDataset

# E2GNN import
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN

# Safe CUDA with DataLoader
mp.set_start_method('spawn', force=True)

# --- Configuration --------------------------------------
device = "cuda:0"
torch.set_default_dtype(torch.float)

NUM_EPOCHS   = 100
BATCH_SIZE   = 64
LR           = 1e-3
ENERGY_W     = 5.0
FORCE_W      = 100.0
HESSIAN_W    = 400.0
GRAD_NORM    = 10.0
EMA_DECAY    = 0.999
WEIGHT_DECAY = 2e-6
PATIENCE     = 5
FACTOR       = 0.8
LR_MIN       = 1e-6
HESS_MAX     = 128  # upper bound for sampling

# Loss functions
loss_e = MSELoss(reduction='sum')
loss_f = MSELoss(reduction='sum')
loss_h = MSELoss(reduction='sum')

# --- Helpers --------------------------------------------
def combine_xyz_files(paths):
    atoms = []
    for p in paths:
        atoms.extend(read(p, ":"))
    return atoms

def combine_target_dicts(paths):
    combined = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        for k in combined:
            combined[k].extend(d[k])
    for k in combined:
        combined[k] = [x.cpu() for x in combined[k]]
    return combined

def compute_sampled_hessian(batch, mdl, loss_fn, max_samples):
    data   = batch.to_data_list()[0]
    single = Batch.from_data_list([data])
    single = move_batch(single, device, torch.float)
    single.pos = single.pos.detach().clone().requires_grad_(True)

    _, forces = mdl(single)
    flat_f    = forces.view(-1)
    n         = flat_f.numel()
    H_gt      = single.hessian.view(n, n)
    idx       = torch.randperm(n, device=flat_f.device)[:max_samples]

    losses = []
    for i in idx:
        go    = torch.zeros_like(flat_f); go[i] = 1.0
        g     = torch.autograd.grad(flat_f, single.pos,
                                    grad_outputs=go,
                                    retain_graph=True,
                                    create_graph=True)[0]
        pred  = -g.view(-1)
        true  = H_gt[i]
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
        sh += compute_sampled_hessian(batch, mdl, loss_h, max_samples=HESS_MAX).item()
    return se, sf, sh

def evaluate_full(loader, mdl, hess_rows):
    mdl.eval()
    e_true = []; e_pred = []
    f_true = []; f_pred = []
    h_true = []; h_pred = []

    for batch in tqdm(loader, desc="Full eval"):
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

            n    = f_flat.numel()
            H_gt = single.hessian.view(n, n)
            idx  = torch.randperm(n, device=device)[:hess_rows]

            for i in idx:
                go    = torch.zeros_like(f_flat); go[i] = 1.0
                g     = torch.autograd.grad(f_flat, single.pos,
                                            grad_outputs=go,
                                            retain_graph=True)[0].view(-1)
                h_pred.extend((-g).cpu().tolist())
                h_true.extend(H_gt[i].cpu().tolist())

    return e_true, e_pred, f_true, f_pred, h_true, h_pred

def plot_corr_unscaled(x_norm, y_norm, title, xlabel, ylabel,
                       scale, mean, filename):
    x = np.asarray(x_norm) * scale + mean
    y = np.asarray(y_norm) * scale + mean

    vmin, vmax = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad        = (vmax - vmin) * 0.05
    lims       = (vmin - pad, vmax + pad)

    mae = mean_absolute_error(y, x)
    mse = mean_squared_error(y, x)

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.3, s=5)
    plt.plot(lims, lims, '--', linewidth=1, color='black')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}\nMAE={mae:.2e}, MSE={mse:.2e}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main -----------------------------------------------
def main():
    # file lists
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
    XYZ_TEST = ["BOTNet-datasets/dataset_3BPA/test_300K.xyz"]
    PT_TEST  = ["BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_300K.pt"]

    # prepare training data
    train_atoms = combine_xyz_files(XYZ_TRAIN)
    train_tgt   = combine_target_dicts(PT_TRAIN)
    all_h       = torch.cat([h.flatten() for h in train_tgt['hessian']])
    hessian_std = all_h.std()

    train_ds     = AtomsDataset(train_atoms, train_tgt, device, hessian_scale=hessian_std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn_e2gnn)
    mean_e, std_e, h_scale = train_ds.energy_mean, train_ds.energy_std, train_ds.hessian_scale

    # prepare validation & test data
    test_atoms = combine_xyz_files(XYZ_TEST)
    test_tgt   = combine_target_dicts(PT_TEST)
    idx = list(range(len(test_atoms)))
    idx_val, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    val_atoms  = [test_atoms[i] for i in idx_val]
    test_atoms = [test_atoms[i] for i in idx_test]
    val_tgt    = {k: [test_tgt[k][i] for i in idx_val] for k in test_tgt}
    test_tgt   = {k: [test_tgt[k][i] for i in idx_test] for k in test_tgt}

    val_ds     = AtomsDataset(val_atoms, val_tgt,       device,
                              energy_mean=mean_e,
                              energy_std=std_e,
                              hessian_scale=h_scale)
    test_ds    = AtomsDataset(test_atoms, test_tgt,     device,
                              energy_mean=mean_e,
                              energy_std=std_e,
                              hessian_scale=h_scale)
    val_loader = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn_e2gnn)
    test_loader= DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn_e2gnn)

    # output directory
    out_dir = "results/e2gnn_hessian_sweep"
    os.makedirs(out_dir, exist_ok=True)

    # hess_steps = [0]+ np.logspace(0, 7, base=2, num=7).astype(int).tolist()
    hess_steps = np.logspace(0, 7, base=2, num=7).astype(int).tolist()

    for hess_rows in hess_steps:
        print(f"\n=== Training with HESS_ROWS = {hess_rows} ===")
        globals()['HESS_ROWS'] = hess_rows

        # reset batch counter & timer
        total_batches = 0
        start_time    = time.time()

        # instantiate model + EMA + optim + scheduler
        model     = E2GNN(hidden_channels=128, num_layers=3, num_rbf=32,
                          cutoff=4.5, max_neighbors=15,
                          use_pbc=False, otf_graph=True,
                          num_elements=9).to(device)
        ema_model = AveragedModel(model,
                                  avg_fn=lambda a,c,n: EMA_DECAY*a + (1-EMA_DECAY)*c)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, 'min', factor=FACTOR, patience=PATIENCE, min_lr=LR_MIN
                )

        # training
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in tqdm(train_loader, desc=f"[H={hess_rows}] Epoch {epoch+1}/{NUM_EPOCHS}"):
                total_batches += 1
                batch = move_batch(batch, device, torch.float)
                e_p, f_p = model(batch)
                e_l = loss_e(e_p, batch.y)
                f_l = loss_f(f_p, batch.force)
                h_l = compute_sampled_hessian(batch, model, loss_h, max_samples=hess_rows)

                loss = ENERGY_W*e_l + FORCE_W*f_l + HESSIAN_W*h_l
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
                opt.step()
                ema_model.update_parameters(model)

            # validation & LR scheduling
            se, sf, sh = evaluate(val_loader, model)
            sched.step((se + sf + sh) / len(val_loader.dataset))

        # compute times
        elapsed = time.time() - start_time
        avg_it_s = total_batches / elapsed

        # save model
        model_path = os.path.join(out_dir,
                                  f"e2gnn_student_supervised_HESSIAN_n{hess_rows}.model")
        torch.save(ema_model.module, model_path)

        # log timing
        timing_file = os.path.join(out_dir, f"{hess_rows}_training_time.txt")
        with open(timing_file, 'w') as f:
            f.write(f"Total time: {elapsed:.2f} s\n")
            f.write(f"Avg it/s:   {avg_it_s:.2f}\n")

        # full evaluation & plots
        e_t, e_p, f_t, f_p, h_t, h_p = evaluate_full(test_loader, ema_model.module, hess_rows)

        plot_corr_unscaled(
            x_norm   = e_p, y_norm   = e_t,
            title    = "Energy Correlation",
            xlabel   = "True Energy (eV)",
            ylabel   = "Predicted Energy (eV)",
            scale    = std_e.item(),
            mean     = mean_e.item(),
            filename = os.path.join(out_dir, f"corr_energy_H{hess_rows}.png")
        )

        plot_corr_unscaled(
            x_norm   = f_p, y_norm   = f_t,
            title    = "Force Correlation",
            xlabel   = "True Force (eV/Å)",
            ylabel   = "Predicted Force (eV/Å)",
            scale    = std_e.item(),
            mean     = 0.0,
            filename = os.path.join(out_dir, f"corr_force_H{hess_rows}.png")
        )

        plot_corr_unscaled(
            x_norm   = h_p, y_norm   = h_t,
            title    = "Hessian Correlation",
            xlabel   = "True Hessian (eV/Å²)",
            ylabel   = "Predicted Hessian (eV/Å²)",
            scale    = h_scale.item(),
            mean     = 0.0,
            filename = os.path.join(out_dir, f"corr_hessian_H{hess_rows}.png")
        )

        print(f"Done H={hess_rows}: {elapsed:.2f}s, {avg_it_s:.2f} it/s")

if __name__ == "__main__":
    main()
