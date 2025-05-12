import os
import sys
import random
import multiprocessing as mp
# combine imports for read
from ase.io import read
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TorchANI imports
import torchani
from torchani.utils import ChemicalSymbolsToInts, hessian as torchani_hessian # renamed to avoid conflict
from utils.torchani_helper import collate_fn


import time
import pandas as pd

# --- Helper Data Handling Classes and Functions (adapted from/for TorchANI) ---
class AtomsDataset(torch.utils.data.Dataset):
    def __init__(self, species_tensor, coordinates_tensor, targets_dict, device):
        self.species = species_tensor.to(device)
        self.coordinates = coordinates_tensor.to(device)
        
        # Store raw targets
        self.energies = torch.tensor([e.item() for e in targets_dict['energy']], dtype=torch.float).to(device)
        self.forces = [f.to(device) for f in targets_dict['forces']]
        self.hessians = [h.to(device) for h in targets_dict['hessian']]

    def __len__(self):
        return self.species.shape[0]

    def __getitem__(self, idx):
        s = self.species[idx]
        c = self.coordinates[idx]
        e = self.energies[idx] 
        f = self.forces[idx]
        h = self.hessians[idx]
        return s, c, e, f, h

def collate_fn_torchani(batch_list):
    species, coordinates, energies, forces, hessians = zip(*batch_list)
    return {
        'species': torch.stack(list(species)),
        'coordinates': torch.stack(list(coordinates)),
        'energy': torch.tensor(list(energies), dtype=torch.float), # Ensure it's a tensor
        'forces': torch.stack(list(forces)),
        'hessian': torch.stack(list(hessians))
    }

def move_batch(batch, device, dtype=torch.float):
      return {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device)
              for k, v in batch.items()}

def read_torchani_xyz(path, species_converter_fn, device):
    atoms_list = read(path, index=':')
    all_symbols = [atoms.get_chemical_symbols() for atoms in atoms_list]
    all_coords = [atoms.get_positions() for atoms in atoms_list]

    all_species_converted = [species_converter_fn(symbols) for symbols in all_symbols]
    # Ensure all species tensors have the same length by padding if necessary,
    # or assume fixed number of atoms as in train_torchani_precompute_hessian.py
    # For simplicity, assuming fixed number of atoms based on training script.
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
        for k in combined:
            combined[k].extend(d[k])
    for k in combined: # Move to CPU for general handling, will be moved to device by Dataset
        combined[k] = [x.cpu() for x in combined[k]]
    return combined

# --- Evaluation Function ---
def evaluate_full(loader, mdl, device, dt_fs, csv_path, hess_rows=128): # Matched HESS_ROWS from training
    """
    Runs through the loader, computes predictions, and logs
    throughput in ns/day based on dt_fs (timestep in fs).
    
    Args:
        loader: DataLoader yielding batches of graphs
        mdl:       the compiled/EMA model
        device:    e.g. "cuda:0"
        dt_fs:     timestep in femtoseconds (e.g. 1.0 for 1 fs)
        csv_path:  path to write CSV with column 'ns_day'
        hess_rows: how many Hessian rows to sample per graph for Hessian evaluation
    Returns:
        (e_true, e_pred, f_true, f_pred, h_true, h_pred) # Added h_true, h_pred
    """
    mdl.eval() # Ensure model is in evaluation mode
    e_true, e_pred = [], []
    f_true, f_pred = [], []
    h_true, h_pred = [], [] # For Hessian components

    n_steps = 0
    t0 = time.time()

    for batch_data in tqdm(loader, desc="Evaluating"):
        batch_data = move_batch(batch_data, device, torch.float)
        
        species_b = batch_data['species']
        coords_b = batch_data['coordinates']
        energies_b = batch_data['energy'] # These are raw energies from AtomsDataset
        forces_b = batch_data['forces']
        hessians_b = batch_data['hessian'] # Raw Hessians

        current_batch_size = species_b.shape[0]
        for i in range(current_batch_size):
            n_steps += 1
            s_i = species_b[i].unsqueeze(0) # (1, num_atoms)
            c_i = coords_b[i].unsqueeze(0).detach().clone().requires_grad_(True) # (1, num_atoms, 3)

            # energy & force
            model_out = mdl((s_i, c_i)) # TorchANI model call
            e_out_val = model_out.energies # This is a scalar tensor
            
            e_true.append(energies_b[i].item())
            e_pred.append(e_out_val.item())

            # Forces: f = -grad(E)
            # Retain graph False for inference, create_graph False
            f_out_val = -torch.autograd.grad(e_out_val.sum(), c_i, create_graph=False, retain_graph=False)[0]
            
            f_pred_flat = f_out_val.view(-1)
            f_true.extend(forces_b[i].view(-1).cpu().tolist())
            f_pred.extend(f_pred_flat.cpu().tolist())

            # Predicted Hessian (full, then sample)
            # H_pred_full is (N_atoms*3, N_atoms*3)
            H_pred_full = torchani_hessian(c_i, energies=e_out_val)[0] # Access the first (and only) item in batch
            
            n_dof = H_pred_full.shape[0]
            H_gt_full = hessians_b[i].view(n_dof, n_dof) # Ground truth Hessian for the i-th sample
            
            idx_sample = torch.randperm(n_dof, device=device)[:hess_rows] # Sample rows
            
            h_pred.extend(H_pred_full[idx_sample].cpu().flatten().tolist()) # Get sampled rows, flatten, and collect
            h_true.extend(H_gt_full[idx_sample].cpu().flatten().tolist())   # Get corresponding GT rows

    wall_seconds = time.time() - t0

    # compute simulated time in nanoseconds
    sim_ns = n_steps * dt_fs * 1e-6  # fs → ns
    # compute throughput in ns/day
    ns_per_day = sim_ns / wall_seconds * 86400.0

    # log to CSV
    df = pd.DataFrame([{'ns_day': ns_per_day}])
    df.to_csv(csv_path, index=False)

    print(f"Processed {n_steps} steps in {wall_seconds:.1f}s → {ns_per_day:.2f} ns/day")

    return e_true, e_pred, f_true, f_pred, h_true, h_pred

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

# --- Main Script ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Good practice for CUDA + multiprocessing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32) # TorchANI typically uses float32

    # TorchANI specific: species converter
    # Using ANI1x species as an example, ensure this matches your trained model
    torchspecies = torchani.models.ANI1x().species 
    species_converter = ChemicalSymbolsToInts(torchspecies)

    # Load TorchANI model
    # This should be the path to the model saved by train_torchani_precompute_hessian.py
    # e.g., "torchani_student_supervised_HESSIAN_mixedT.model"
    model_path = "torchani_student_supervised_HESSIAN.model" 
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please ensure the trained TorchANI model is available at this path.")
        sys.exit(1)
        
    loaded_model = torch.load(model_path, map_location=device)
    loaded_model.to(device)
    loaded_model.eval()

    # Define paths for test data (adjust as per your TorchANI dataset)
    # These should match the kind of data your TorchANI model was trained/validated on.
    XYZ_TEST = [
        "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/test_300K.xyz", # Example from train_torchani
    ]
    PT_TEST = [
        "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_test_300K.pt", # Example
    ]

    # Create output directory
    output_dir = "results_for_analysis/torchani_hessian"
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    test_species_tensor, test_coords_tensor = combine_torchani_xyz(XYZ_TEST, species_converter, device)
    test_tgt_dict = combine_target_dicts(PT_TEST)

    # Create dataset and dataloader
    # For inference, we don't need training-set derived mean/std for normalization of targets,
    # as the TorchANI model (with EnergyShifter) predicts absolute energies.
    # We compare absolute predicted energies with absolute true energies.
    test_ds = AtomsDataset(test_species_tensor, test_coords_tensor, test_tgt_dict, device)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0) # Batch size can be adjusted

    # Evaluate model
    test_e_true, test_e_pred, test_f_true, test_f_pred, test_h_true, test_h_pred = evaluate_full(
        test_loader, loaded_model, device,
        dt_fs=1.0, # Timestep for ns/day calculation, adjust if needed
        csv_path=os.path.join(output_dir, "torchani_hessian_inference.csv")
    )

    # Plot correlations
    plot_corr(test_e_true, test_e_pred, "Energy Correlation (TorchANI)", "True Energy", "Predicted Energy", os.path.join(output_dir, "correlation_plot_energy.png"))
    plot_corr(test_f_true, test_f_pred, "Force Correlation (TorchANI)", "True Force", "Predicted Force", os.path.join(output_dir, "correlation_plot_force.png"))
    plot_corr(test_h_true, test_h_pred, "Hessian Sampled Correlation (TorchANI)", "True Hessian Components", "Predicted Hessian Components", os.path.join(output_dir, "correlation_plot_hessian.png"))

    print(f"Inference complete. Results saved to {output_dir}")