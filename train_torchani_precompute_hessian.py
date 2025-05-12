import os
import sys
import random
import multiprocessing as mp
import numpy as np
# combine imports for read
from ase.io import read
import torch
from torch import compile as torch_compile
from torch.nn import MSELoss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# torchani imports
import torchani
from torchani import AEVComputer
from torchani.utils import ChemicalSymbolsToInts, hessian
#from torchani.data import collate_fn as collate_fn_torchani
# Add required imports for custom model construction
from torchani.nn import ANIModel
from torchani.nn import Sequential
from utils.torchani_helper import build_nn, init_normal, AtomsDataset, collate_fn
#from loaders.loaders import AtomsDataset 


# Ensure 'spawn' start method for safe CUDA in DataLoader workers
mp.set_start_method('spawn', force=True)

device = "cuda:1"
torch.set_default_dtype(torch.float)

# --- Hyperparameters --------------------------------------
NUM_EPOCHS, BATCH_SIZE = 100, 64
LR, ENERGY_W, FORCE_W, HESSIAN_W = 1e-3, 5.0, 100.0, 400.0
GRAD_NORM, EMA_DECAY, WEIGHT_DECAY = 10.0, 0.999, 2e-6
PATIENCE, FACTOR, LR_MIN, HESS_ROWS = 5, 0.8, 1e-6, 128

# TorchANI model
torchspecies = torchani.models.ANI1x().species
species_converter = ChemicalSymbolsToInts(torchspecies)

# AEVComputer
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

ani_network = ANIModel(
    build_nn()
)
# Get the pretrained energy shift from ANI1x to calibrate outputs
pretrained = torchani.models.ANI1x()
energy_shifter = pretrained.energy_shifter

# Loss functions
loss_e = MSELoss(reduction='sum')
loss_f = MSELoss(reduction='sum')
loss_h = MSELoss(reduction='sum')

# --- Helper Functions ------------------------------------

def read_torchani_xyz(path):
    atoms_list = read(path, index=':')
    all_symbols = [atoms.get_chemical_symbols() for atoms in atoms_list]
    all_coords = [atoms.get_positions() for atoms in atoms_list]  # list of (27, 3) arrays

    all_species = [species_converter(symbols) for symbols in all_symbols]
    species_tensor = torch.stack(all_species).to(device)  # (500, 27)

    # Use np.stack to keep the shape (500, 27, 3)
    coords_tensor = torch.tensor(np.stack(all_coords), dtype=torch.float32, device=device)  # (500, 27, 3)
    return species_tensor, coords_tensor


def plot_corr(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.3, s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def combine_torchani_xyz(paths):
    Ss, Cs = [], []
    for p in paths:
        s, c = read_torchani_xyz(p)
        Ss.append(s); Cs.append(c)
    return torch.cat(Ss, dim=0), torch.cat(Cs, dim=0)


def combine_target_dicts(paths):
    # 1) Initialize empty lists for each target type
    combined = {"energy": [], "forces": [], "hessian": []}

    # 2) Loop over every .pt file you passed in
    for p in paths:
        d = torch.load(p)            # load one dict, e.g. {"energy":[...], "forces":[...], ...}
        for k in combined:           # for each of the three keys
            combined[k].extend(d[k]) #   append that file’s list onto the master list

    # 3) Move any GPU tensors back to CPU for safe indexing/storage
    #    (so you don’t carry around CUDA tensors if you later want to serialize or slice)
    for k in combined:
        combined[k] = [x.cpu() for x in combined[k]]

    # 4) Return a dict of three Python lists:
    #    - combined["energy"] is a list of scalars (0-D tensors)
    #    - combined["forces"] is a list of (natoms×3) tensors
    #    - combined["hessian"] is a list of (n_dof×n_dof) tensors
    return combined


def move_batch(batch, device, dtype=torch.float):
      return {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device)
              for k, v in batch.items()}



def compute_sampled_hessian(batch, mdl, loss_fn, max_samples=HESS_ROWS):
    species = batch['species']
    coords = batch['coordinates']
    hessians = batch['hessian']
    batch_size = species.shape[0]
    losses = []

    for i in range(batch_size):
        s = species[i].unsqueeze(0)
        c = coords[i].unsqueeze(0).detach().clone().requires_grad_(True)
        h_gt = hessians[i]
        n = h_gt.shape[0]

        energies = mdl((s, c)).energies
        H_pred_full = hessian(c, energies=energies)[0]
        #assert H_pred_full.shape == h_gt.shape, f"Hessian dimension mismatch: {H_pred_full.shape} vs {h_gt.shape}"

        idx = torch.randperm(n, device=c.device)[:max_samples]
        for j in idx:
            pred_row = H_pred_full[j]
            true_row = h_gt[j]
            losses.append(loss_fn(pred_row, true_row))

    return sum(losses) / len(losses) if losses else torch.tensor(0., device=coords.device)



def evaluate(loader, mdl):
    mdl.eval()
    se = sf = sh = 0.0
    for batch in tqdm(loader, desc="Evaluating"):
        b = move_batch(batch, device, torch.float)
        species_batch = b['species']
        coords_batch = b['coordinates'].detach().clone().requires_grad_(True)
        e_p = mdl((species_batch, coords_batch)).energies
        f_p = -torch.autograd.grad(e_p.sum(), coords_batch, create_graph=True, retain_graph=True)[0]
        se += loss_e(e_p, b['energy'].squeeze(-1)).item()
        sf += loss_f(f_p, b['forces']).item()
        sh += compute_sampled_hessian(b, mdl, loss_h).item()
    return se, sf, sh


def evaluate_full(loader, mdl, device, hess_rows=HESS_ROWS):
    mdl.eval()
    e_true, e_pred = [], []
    f_true, f_pred = [], []
    h_true, h_pred = [], []

    for batch in loader:
        batch = move_batch(batch, device, torch.float)
        species = batch['species']
        coords = batch['coordinates']
        energies = batch['energy']
        forces = batch['forces']
        hessians = batch['hessian']
        batch_size = species.shape[0]

        for i in range(batch_size):
            s = species[i].unsqueeze(0)
            c = coords[i].unsqueeze(0).detach().clone().requires_grad_(True)
            e_t = energies[i].item() if energies[i].numel() == 1 else energies[i].squeeze().item()
            f_t = forces[i].view(-1).cpu().tolist()
            h_t = hessians[i].view(-1).cpu().tolist()

            # Model prediction
            e_out = mdl((s, c)).energies
            e_p = e_out.item()
            f_out = -torch.autograd.grad(e_out.sum(), c, create_graph=True, retain_graph=True)[0]
            f_p = f_out.view(-1).cpu().tolist()

            # Hessian: sample rows
            n = f_out.numel()
            h_gt = hessians[i].view(n, n)
            idx = torch.randperm(n, device=device)[:hess_rows]
            for j in idx:
                go = torch.zeros_like(f_out.view(-1))
                go[j] = 1.0
                g = torch.autograd.grad(f_out.view(-1), c, grad_outputs=go, retain_graph=True)[0].view(-1)
                h_pred.append(g[j].item())
                h_true.append(h_gt[j].cpu().item())

            # Collect energy and force
            e_true.append(e_t)
            e_pred.append(e_p)
            f_true.extend(f_t)
            f_pred.extend(f_p)

    return e_true, e_pred, f_true, f_pred, h_true, h_pred


def main():

    # Data loading
    XYZ_TRAIN = [
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/train_300K.xyz",
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/train_mixedT.xyz",
         "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/test_dih.xyz",
    ]

    #XYZ_TRAIN = "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/train_300K.xyz"

    PT_TRAIN = [
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_train_300K.pt",
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_train_mixedT.pt",
         "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_test_dih.pt",
    ]
    

    XYZ_TEST = [
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/test_300K.xyz",
    #     "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/test_mixedT.xyz",
         "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/test_dih.xyz",
    ]

    PT_TEST = [
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_test_300K.pt",
    #    "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_test_mixedT.pt",
        "BOTNet-datasets/precomputed_3BPA/dataset_3BPA/precomputed_training_data_test_dih.pt",

    ]

    train_species, train_coords = combine_torchani_xyz(XYZ_TRAIN)  # (N, 27) for species and (N, 27, 3) for coordinates
    train_tgt = combine_target_dicts(PT_TRAIN) # (N, 1) for energy, (N, 3) for forces, (N, 27*27) for hessian
    
    #print(train_tgt["energy"][:3])
    #print(train_tgt["forces"][:3])
    #print(train_tgt["hessian"][:3])
    
    all_h = torch.cat([h.flatten() for h in train_tgt['hessian']])
    hessian_std = all_h.std()
    #print(f"train_species: {train_species.shape}, train_coords: {train_coords.shape}")
    train_ds = AtomsDataset(train_species, train_coords, train_tgt, device, hessian_scale=hessian_std)

    print("Length of train_ds:", len(train_ds))
    print("TRAIN SAMPLE:")
    sample_train = train_ds[0]
    print(type(sample_train), len(sample_train))
    for i, x in enumerate(sample_train):
        print(f"Item {i}: shape_train {x.shape}, dtype_train {x.dtype}, device_train {x.device}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)#,num_workers=8) 
    mean_e, std_e, h_scale = train_ds.energy_mean, train_ds.energy_std, train_ds.hessian_scale
    
    try:
        batch_train = next(iter(train_loader))
        print(type(batch_train))
        if isinstance(batch_train, dict):
            print(batch_train.keys())
        elif isinstance(batch_train, tuple):
            print(len(batch_train))
            for i, x in enumerate(batch_train):
                print(f"Batch item {i}: shape {x.shape}, dtype {x.dtype}, device {x.device}")
    except Exception as e:
        print(f"Error: {e}")

    test_species, test_coords = combine_torchani_xyz(XYZ_TEST)
    test_tgt   = combine_target_dicts(PT_TEST)

    # 2) split 50/50 by index
    indices      = list(range(len(test_species)))
    idx_val, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    # 3) slice into val vs. test
    val_species, val_coords = [test_species[i] for i in idx_val], [test_coords[i] for i in idx_val]
    test_species, test_coords = [test_species[i] for i in idx_test], [test_coords[i] for i in idx_test]

    val_tgt = { k: [test_tgt[k][i] for i in idx_val]   for k in test_tgt }
    test_tgt= { k: [test_tgt[k][i] for i in idx_test]  for k in test_tgt }

    # 4) build datasets & loaders
    val_ds    = AtomsDataset(val_species, val_coords, val_tgt, device, hessian_scale=hessian_std)
    test_ds   = AtomsDataset(test_species, test_coords, test_tgt, device, hessian_scale=hessian_std)
    
    print(f"Total test samples: {len(test_ds)}")
    print(f"Total val samples: {len(val_ds)}")

    # Print a sample from val and test
    sample_val = val_ds[0]
    sample_test = test_ds[0]

    print("VAL SAMPLE:")
    for i, x in enumerate(sample_val):
        print(f"  Item {i}: shape_val {x.shape}, dtype_val {x.dtype}, device_val {x.device}")

    print("TEST SAMPLE:")
    for i, x in enumerate(sample_test):
        print(f"  Item {i}: shape_test {x.shape}, dtype_test {x.dtype}, device_test {x.device}")

    # Optionally, print the indices used for splitting
    #print("Validation indices (first 10):", idx_val[:10])
    #print("Test indices (first 10):", idx_test[:10])

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # construct model
    # Combine steps: AEV → element MLP → energy shift
    model = Sequential(
        aev_computer,
        ani_network,
        energy_shifter
    ).to(device).to(torch.float)

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
            species_batch = batch['species']
            coords_batch = batch['coordinates'].requires_grad_(True)
            e_p = model((species_batch, coords_batch)).energies
            f_p = -torch.autograd.grad(e_p.sum(), coords_batch, create_graph=True, retain_graph=True)[0]
            e_l = loss_e(e_p, batch['energy'].squeeze(-1))
            f_l = loss_f(f_p, batch['forces'])
            h_l = compute_sampled_hessian(batch, model, loss_h)
            loss = ENERGY_W*e_l + FORCE_W*f_l + HESSIAN_W*h_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            opt.step()
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
    torch.save(ema_model.module, "torchani_student_supervised_HESSIAN_mixedT.model")
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