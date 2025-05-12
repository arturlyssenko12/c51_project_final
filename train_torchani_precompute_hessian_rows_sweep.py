#!/usr/bin/env python3
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from ase.io import read

# TorchANI imports
import torchani
from torchani import AEVComputer
from torchani.utils import ChemicalSymbolsToInts, hessian as torchani_hessian # renamed
from torchani.nn import ANIModel, Sequential
from utils.torchani_helper import collate_fn, build_nn

# Safe CUDA with DataLoader
mp.set_start_method('spawn', force=True)

# --- Configuration --------------------------------------
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

NUM_EPOCHS   = 100 # As in E2GNN sweep
BATCH_SIZE   = 64  # As in E2GNN sweep
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
# HESS_ROWS will be set by the sweep loop

# Loss functions
loss_e = MSELoss(reduction='sum')
loss_f = MSELoss(reduction='sum')
loss_h = MSELoss(reduction='sum')

# --- TorchANI Model and Data Helpers (adapted from train_torchani_precompute_hessian.py) ---

torchspecies = torchani.models.ANI1x().species # Example, ensure matches data
species_converter = ChemicalSymbolsToInts(torchspecies)

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = len(torchspecies)
aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

ani_network = ANIModel(build_nn())
energy_shifter = torchani.models.ANI1x().energy_shifter # Get a standard energy shifter

class AtomsDataset(Dataset):
    def __init__(self, species_tensor, coordinates_tensor, targets_dict, device, hessian_scale=1.0):
        self.species = species_tensor.to(device)
        self.coordinates = coordinates_tensor.to(device)
        self.energies = torch.tensor([e.item() for e in targets_dict['energy']], dtype=torch.float).to(device)
        self.forces = [f.to(device) for f in targets_dict['forces']]
        self.hessians = [h.to(device) for h in targets_dict['hessian']]

        self.energy_mean = self.energies.mean()
        self.energy_std = self.energies.std()
        if self.energy_std == 0: self.energy_std = torch.tensor(1.0, device=device)
        self.hessian_scale = torch.tensor(hessian_scale, dtype=torch.float, device=device)

    def __len__(self):
        return self.species.shape[0]

    def __getitem__(self, idx):
        return self.species[idx], self.coordinates[idx], self.energies[idx], self.forces[idx], self.hessians[idx]

def collate_fn_torchani(batch_list):
    species, coordinates, energies, forces, hessians = zip(*batch_list)
    return {
        'species': torch.stack(list(species)),
        'coordinates': torch.stack(list(coordinates)),
        'energy': torch.tensor(list(energies), dtype=torch.float),
        'forces': torch.stack(list(forces)),
        'hessian': torch.stack(list(hessians))
    }

def read_torchani_xyz(path, species_converter_fn, device):
    atoms_list = read(path, index=':')
    all_symbols = [atoms.get_chemical_symbols() for atoms in atoms_list]
    all_coords = [atoms.get_positions() for atoms in atoms_list]
    all_species_converted = [species_converter_fn(symbols) for symbols in all_symbols]
    species_tensor = torch.stack(all_species_converted).to(device)
    coords_tensor = torch.tensor(np.stack(all_coords), dtype=torch.float32, device=device)
    return species_tensor, coords_tensor

def combine_torchani_xyz(paths, species_converter_fn, device):
    Ss, Cs = [], []
    for p in paths:
        s, c = read_torchani_xyz(p, species_converter_fn, device)
        Ss.append(s); Cs.append(c)
    return torch.cat(Ss, dim=0), torch.cat(Cs, dim=0)

def combine_target_dicts(paths):
    combined = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        for k in combined: combined[k].extend(d[k])
    for k in combined: combined[k] = [x.cpu() for x in combined[k]] # Move to CPU first
    return combined

def move_batch(batch, device, dtype=torch.float):
      return {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device)
              for k, v in batch.items()}

def compute_sampled_hessian(batch_data, mdl, loss_fn, max_samples):
    # batch_data is already on device
    species_b = batch_data['species']
    coords_b = batch_data['coordinates']
    hessians_gt_b = batch_data['hessian']
    
    batch_size = species_b.shape[0]
    losses = []

    for i in range(batch_size):
        s_i = species_b[i].unsqueeze(0) # (1, num_atoms)
        c_i = coords_b[i].unsqueeze(0).detach().clone().requires_grad_(True) # (1, num_atoms, 3)
        h_gt_i = hessians_gt_b[i] # (N_dof, N_dof)
        n_dof = h_gt_i.shape[0]

        energies_pred = mdl((s_i, c_i)).energies
        H_pred_full = torchani_hessian(c_i, energies=energies_pred)[0] # Get (N_dof, N_dof) for the first (only) sample

        idx_sample = torch.randperm(n_dof, device=c_i.device)[:max_samples]
        
        pred_rows = H_pred_full[idx_sample]
        true_rows = h_gt_i[idx_sample]
        losses.append(loss_fn(pred_rows, true_rows))

    return sum(losses) / len(losses) if losses else torch.tensor(0., device=coords_b.device)

def evaluate(loader, mdl, current_hess_rows): # Added current_hess_rows
    mdl.eval()
    se = sf = sh = 0.0
    for batch_data in loader: # Not using tqdm here to match E2GNN sweep
        b = move_batch(batch_data, device, torch.float)
        species_batch = b['species']
        coords_batch = b['coordinates'].detach().clone().requires_grad_(True)
        
        e_p = mdl((species_batch, coords_batch)).energies
        # Forces: f = -grad(E)
        f_p = -torch.autograd.grad(e_p.sum(), coords_batch, create_graph=False, retain_graph=False)[0]
        
        se += loss_e(e_p, b['energy'].squeeze(-1)).item() # Assuming energy is (batchsize, 1)
        sf += loss_f(f_p, b['forces']).item()
        # For Hessian loss in validation, use the same number of rows as in training for consistency
        sh += compute_sampled_hessian(b, mdl, loss_h, max_samples=current_hess_rows).item()
    return se, sf, sh

def evaluate_full(loader, mdl, current_hess_rows): # Added current_hess_rows
    mdl.eval()
    e_true, e_pred = [], []
    f_true, f_pred = [], []
    h_true, h_pred = [], []

    for batch_data in tqdm(loader, desc="Full eval"):
        batch_data = move_batch(batch_data, device, torch.float)
        species_b = batch_data['species']
        coords_b = batch_data['coordinates']
        energies_gt_b = batch_data['energy']
        forces_gt_b = batch_data['forces']
        hessians_gt_b = batch_data['hessian']
        
        current_batch_size = species_b.shape[0]
        for i in range(current_batch_size):
            s_i = species_b[i].unsqueeze(0)
            c_i = coords_b[i].unsqueeze(0).detach().clone().requires_grad_(True)

            # Energy and Force prediction
            model_out = mdl((s_i, c_i))
            e_p_val = model_out.energies
            f_p_val = -torch.autograd.grad(e_p_val.sum(), c_i, create_graph=True, retain_graph=True)[0]

            e_true.append(energies_gt_b[i].item())
            e_pred.append(e_p_val.item())
            f_true.extend(forces_gt_b[i].view(-1).cpu().tolist())
            f_pred.extend(f_p_val.view(-1).cpu().tolist())

            # Hessian prediction
            H_pred_full = torchani_hessian(c_i, energies=e_p_val)[0]
            H_gt_full = hessians_gt_b[i] # Already (N_dof, N_dof)
            n_dof = H_gt_full.shape[0]
            
            idx_sample = torch.randperm(n_dof, device=device)[:current_hess_rows]
            
            h_pred.extend(H_pred_full[idx_sample].cpu().flatten().tolist())
            h_true.extend(H_gt_full[idx_sample].cpu().flatten().tolist())

    return e_true, e_pred, f_true, f_pred, h_true, h_pred

def plot_corr(x, y, title, xlabel, ylabel, filename): # Simpler plot_corr
    x = np.asarray(x)
    y = np.asarray(y)
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    margin = (vmax - vmin) * 0.05
    lims = (vmin - margin, vmax + margin)
    mae = mean_absolute_error(y, x)
    mse = mean_squared_error(y, x)

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.3, s=5)
    plt.plot(lims, lims, linestyle='--', linewidth=1, color='black')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}\nMAE={mae:.3e}, MSE={mse:.3e}") # Adjusted MAE/MSE format
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main -----------------------------------------------
def main():
    global HESS_ROWS # Allow sweep loop to modify this global for helper functions

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

    train_species, train_coords = combine_torchani_xyz(XYZ_TRAIN, species_converter, device)
    train_tgt = combine_target_dicts(PT_TRAIN)
    all_h = torch.cat([h.flatten() for h in train_tgt['hessian']])
    hessian_std_val = all_h.std().item() # scalar value

    train_ds = AtomsDataset(train_species, train_coords, train_tgt, device, hessian_scale=hessian_std_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # For val/test, use the same mean/std/scale as training for consistency if any normalization were applied
    # Here, dataset stores absolute values, so mean/std are informational.
    energy_mean_ref = train_ds.energy_mean.item()
    energy_std_ref = train_ds.energy_std.item()
    hessian_scale_ref = train_ds.hessian_scale.item()

    test_species_all, test_coords_all = combine_torchani_xyz(XYZ_TEST, species_converter, device)
    test_tgt_all = combine_target_dicts(PT_TEST)
    
    indices = list(range(len(test_species_all)))
    idx_val, idx_test = train_test_split(indices, test_size=0.5, random_state=42) # 50/50 split for val/test from "test" data

    val_species = test_species_all[idx_val]
    val_coords = test_coords_all[idx_val]
    test_species = test_species_all[idx_test]
    test_coords = test_coords_all[idx_test]

    val_tgt = {k: [test_tgt_all[k][i] for i in idx_val] for k in test_tgt_all}
    test_tgt_final = {k: [test_tgt_all[k][i] for i in idx_test] for k in test_tgt_all}

    val_ds = AtomsDataset(val_species, val_coords, val_tgt, device, hessian_scale=hessian_std_val)
    test_ds = AtomsDataset(test_species, test_coords, test_tgt_final, device, hessian_scale=hessian_std_val)
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_torchani, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_torchani, num_workers=0)

    out_dir = "results/torchani_hessian_sweep"
    os.makedirs(out_dir, exist_ok=True)

    hess_steps = [0] + np.logspace(0, 7, base=2, num=7).astype(int).tolist() 

    for current_hess_rows_val in hess_steps:
        HESS_ROWS = current_hess_rows_val # Set global for helpers
        print(f"\n=== Training with HESS_ROWS = {HESS_ROWS} ===")

        total_batches = 0
        start_time = time.time()

        model = Sequential(aev_computer, ani_network, energy_shifter).to(device).to(torch.float)
        ema_model = AveragedModel(model, avg_fn=lambda a,c,n: EMA_DECAY*a + (1-EMA_DECAY)*c)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=FACTOR, patience=PATIENCE, min_lr=LR_MIN)

        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in tqdm(train_loader, desc=f"[H={HESS_ROWS}] Epoch {epoch+1}/{NUM_EPOCHS}"):
                total_batches += 1
                batch = move_batch(batch, device, torch.float)
                coords_batch_grad = batch['coordinates'].requires_grad_(True)
                
                e_p = model((batch['species'], coords_batch_grad)).energies
                f_p = -torch.autograd.grad(e_p.sum(), coords_batch_grad, create_graph=True, retain_graph=True)[0]
                
                e_l = loss_e(e_p, batch['energy'].squeeze(-1))
                f_l = loss_f(f_p, batch['forces'])
                
                # Pass batch with coords_batch_grad for Hessian computation if needed
                # For compute_sampled_hessian, it re-enables grad on its own copy.
                h_l = compute_sampled_hessian(batch, model, loss_h, max_samples=HESS_ROWS) if HESS_ROWS > 0 else torch.tensor(0.0, device=device)

                loss = ENERGY_W*e_l + FORCE_W*f_l + HESSIAN_W*h_l
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
                opt.step()
                ema_model.update_parameters(model)

            # se, sf, sh = evaluate(val_loader, ema_model.module, HESS_ROWS) # Pass HESS_ROWS
            # val_loss_total = (se + sf + sh) / len(val_loader.dataset)
            # sched.step(val_loss_total)
            sched.step(loss)

        elapsed = time.time() - start_time
        avg_it_s = total_batches / elapsed if elapsed > 0 else 0

        model_path = os.path.join(out_dir, f"torchani_student_H{HESS_ROWS}.model")
        torch.save(ema_model.module.state_dict(), model_path) # Save state_dict

        timing_file = os.path.join(out_dir, f"H{HESS_ROWS}_training_time.txt")
        with open(timing_file, 'w') as f:
            f.write(f"Total time: {elapsed:.2f} s\nAvg it/s:   {avg_it_s:.2f}\n")

        e_t, e_p, f_t, f_p, h_t, h_p = evaluate_full(test_loader, ema_model.module, HESS_ROWS)

        plot_corr(e_t, e_p, "Energy Correlation", "True Energy (eV)", "Predicted Energy (eV)",
                  os.path.join(out_dir, f"corr_energy_H{HESS_ROWS}.png"))
        plot_corr(f_t, f_p, "Force Correlation", "True Force (eV/Å)", "Predicted Force (eV/Å)",
                  os.path.join(out_dir, f"corr_force_H{HESS_ROWS}.png"))
        if HESS_ROWS > 0:
            plot_corr(h_t, h_p, "Hessian Correlation", "True Hessian (eV/Å²)", "Predicted Hessian (eV/Å²)",
                      os.path.join(out_dir, f"corr_hessian_H{HESS_ROWS}.png"))

        print(f"Done H={HESS_ROWS}: {elapsed:.2f}s, {avg_it_s:.2f} it/s")

if __name__ == "__main__":
    main()