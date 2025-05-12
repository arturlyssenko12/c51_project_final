import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io import read
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import random
import datetime

import torchani 
from torchani import AEVComputer
from torchani.utils import ChemicalSymbolsToInts, hessian
from torchani.data import collate_fn as collate_fn_torchani
from utils.torchani_helper import build_nn, init_normal, XYZ, species_to_tensor

from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric

# ------------------- Config -------------------
TEACHER_PATH = "MACE-OFF23_large.model"
XYZ_PATH = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
ENERGY_WEIGHT = 5.
FORCE_WEIGHT = 100.0
HESSIAN_WEIGHT = 400.0
HESSIAN_ROWS = 1
EMA_DECAY = 0.999
GRAD_CLIP_NORM = 5.0

# Results dir
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ---------------- Torchani Config ----------------

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)

# needed to convert species to indices
num_species = 4

aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

print(f"aev_computer: {aev_computer}")

layers = torchani.ANIModel(build_nn())        

# Pretrained ANI1x self-energies 
ani1x = torchani.models.ANI1x()
shifter = ani1x.energy_shifter

# ------------------- Utilities -------------------
def move_batch(batch, device, dtype=torch.float):
    batch = batch.to(device)
    for attr in ['x', 'positions', 'edge_attr', 'node_attrs']:
        if hasattr(batch, attr):
            val = getattr(batch, attr)
            if isinstance(val, torch.Tensor):
                setattr(batch, attr, val.to(dtype))
    return batch

def finite_difference_hessian_rows_batched(forces_fn, pos, indices, epsilon=1e-3):
    pos = pos.detach()
    n_atoms = pos.size(0)
    n_coords = n_atoms * 3

    eye = torch.eye(n_coords, device=pos.device, dtype=pos.dtype)[indices]
    perturb = epsilon * eye.view(-1, n_atoms, 3)

    pos_plus  = pos.unsqueeze(0) + perturb
    pos_minus = pos.unsqueeze(0) - perturb

    all_pos = torch.cat([pos_plus, pos_minus], dim=0)
    all_forces = torch.stack([forces_fn(p) for p in all_pos])

    f_plus, f_minus = all_forces.chunk(2, dim=0)
    hessian_rows = ((f_plus - f_minus) / (2 * epsilon)).view(len(indices), -1)

    return [row for row in hessian_rows]

# ------------------- AtomicData for MACE helper -------------------
def create_atomic_data_list(atoms_list, model, description):
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    head_name = getattr(model, "available_heads", ["Default"])[0]
    data_list = []
    for atoms in atoms_list:
        config = config_from_atoms(
            atoms,
            key_specification=KeySpecification(info_keys={}, arrays_keys={}),
            head_name=head_name,
        )
        data = AtomicData.from_config(
            config=config,
            z_table=z_table,
            cutoff=cutoff,
            heads=[head_name],
        )
        data_list.append(data)
    print(f"Created {len(data_list)} samples for {description}")
    return data_list

#def create_e2gnn_data_list(atoms_list, description):
#    a2g = AtomsToGraphs(
#        max_neigh=50,
#        radius=5,
#        r_energy=False,
#        r_forces=False,
#        r_fixed=True,
#        r_distances=False,
#        r_edges=False,
#    )
#    data_list = []
#    dummy_forces = torch.tensor([0.0, 0.0, 0.0])
#    for atoms in atoms_list:
#        data = a2g.convert(atoms)
#        data.y = 0
#        data.force = dummy_forces
#        data_list.append(data)
#    print(f"Created {len(data_list)} samples for {description}")
#    return data_list

#def collate_fn_e2gnn(batch):
#    return Batch.from_data_list(batch)

# ------------------- Load Models -------------------
teacher = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.to(device).float()
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False
#teacher_dtype = next(teacher.parameters()).dtype
teacher_dtype = torch.float

#student = E2GNN(
#    hidden_channels=512,
#    num_layers=4,
#    num_rbf=128,
#    cutoff=6.0,
#    max_neighbors=20,
#    use_pbc=False,
#    otf_graph=True
#).to(device=device, dtype=torch.float).train()

student = torchani.nn.Sequential(
    aev_computer, 
    layers,
    shifter
).to(device=device).train()

ema_weights = {
    name: param.clone().detach()
    for name, param in student.named_parameters()
    if param.requires_grad
}

# ------------------- Optimizer -------------------

optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=2e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-6)
loss_fn_energy = nn.MSELoss()
loss_fn_force = nn.L1Loss()

# ------------------- Load Data -------------------
print(f"Loading atoms from {XYZ_PATH}")
atoms_list = read(XYZ_PATH, ":")

teacher_data = create_atomic_data_list(atoms_list, teacher, "teacher")
student_data = XYZ(XYZ_PATH, device=device)

teacher_loader = torch_geometric.dataloader.DataLoader(
    teacher_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
student_loader = DataLoader(
    student_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn_torchani
)

min_len = min(len(teacher_loader), len(student_loader))

# ------------------- Training -------------------
losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_hessian_loss = 0.0

    loop = tqdm(
        zip(teacher_loader, student_loader),
        total=min_len,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for i, (batch_teacher, batch_student) in enumerate(loop):
        if i >= min_len:
            break

        # 1) Prepare teacher batch
        batch_teacher = move_batch(batch_teacher, device, dtype=teacher_dtype)
        
        # 2) Prepare student batch
        species_batch = batch_student['species'].to(device)
        coords_batch  = batch_student['coordinates'].to(device).float().requires_grad_(True)

        # 3) Teacher forward
        batch_teacher.positions.requires_grad_(False)
        #data_list = batch_teacher.to_data_list() 
        #batch_student.positions = batch_teacher.positions.detach().clone()

        teacher_out = teacher(batch_teacher.to_dict())
        target_energy, target_forces = (
            teacher_out["energy"].detach().to(dtype=torch.float),
            teacher_out["forces"].detach().to(dtype=torch.float),
        )

        # 4) Student forward (energy only, forces via FD below)
        _, predicted_energy = student((species_batch, coords_batch))
        predicted_energy = predicted_energy.float()

        # 5) Per-molecule FD Hessian and force loss
        batch_hessian_losses = []
        batch_force_losses = []
        batch_pred_forces = []
        batch_target_forces = []
        for mol_idx in range(coords_batch.shape[0]):
            mol_species = species_batch[mol_idx].unsqueeze(0)
            mol_coords = coords_batch[mol_idx].detach()  # shape: (num_atoms, 3)
            n_atoms = mol_coords.shape[0]
            n_coords = 3 * n_atoms
            idxs = random.sample(range(n_coords), min(HESSIAN_ROWS, n_coords))

            # --- Teacher force function ---
            def teacher_force_fn(r):
                # r: (atoms, 3)
                batch_teacher.positions = r
                # print(f"teacher r.shape: {batch_teacher.positions.shape}")
                out = teacher(batch_teacher.to_dict())
                return out["forces"].detach()
                
                # r: (num_atoms, 3)
                # Identify which nodes in the batch belong to this molecule
                #mask = batch_teacher.batch == mol_idx
                #assert r.shape == batch_teacher.positions[mask].shape, f"Shape mismatch: r {r.shape} vs mask {batch_teacher.positions[mask].shape}"
                #bt_copy = batch_teacher.clone()
                # Assign the new coordinates to all atoms in this molecule
                #bt_copy.positions[mask] = r
                # #return out["forces"][mask].detach()

            # --- Student force function ---
            def student_force_fn(r):
                #print(f"student r.shape: {r.shape}")
                r = r.unsqueeze(0).requires_grad_(True)
                f = -torch.autograd.grad(
                    student((mol_species, r))[1].sum(), r, create_graph=True
                )[0]
                return f.squeeze(0).detach()

            # --- FD Hessian rows and force at r0 ---
            def fd_forces_and_hess_rows(force_fn, pos, indices, epsilon=1e-3):
                pos = pos.detach()
                n_atoms = pos.shape[0]
                n_coords = n_atoms * 3
                eye = torch.eye(n_coords, device=pos.device, dtype=pos.dtype)[indices]
                perturb = epsilon * eye.view(-1, n_atoms, 3)
                pos_list = [pos] + [pos + p for p in perturb] + [pos - p for p in perturb]
                forces_list = [force_fn(p) for p in pos_list]
                f0 = forces_list[0].reshape(-1, 3)
                f_plus = torch.stack(forces_list[1:1+len(indices)])
                f_minus = torch.stack(forces_list[1+len(indices):])
                hess_rows = ((f_plus - f_minus) / (2 * epsilon)).view(len(indices), -1)
                return f0, hess_rows

            # Teacher FD
            t_force0, t_hess_rows = fd_forces_and_hess_rows(teacher_force_fn, mol_coords, idxs)
            # Student FD
            s_force0, s_hess_rows = fd_forces_and_hess_rows(student_force_fn, mol_coords, idxs)

            # Force loss (L1)
            batch_pred_forces.append(s_force0)
            batch_target_forces.append(t_force0)
            batch_force_losses.append(loss_fn_force(s_force0, t_force0))
            # Hessian loss (MSE)
            #mol_hess_loss = sum(
            #    nn.functional.mse_loss(sr, tr) for sr, tr in zip(s_hess_rows, t_hess_rows)
            #) / len(idxs)
            mol_hess_loss = sum(
                nn.functional.mse_loss(sr, tr, reduction='mean') / sr.numel() for sr, tr in zip(s_hess_rows, t_hess_rows)
            ) / len(idxs)
            batch_hessian_losses.append(mol_hess_loss)

        # Stack and average losses
        force_loss = torch.stack(batch_force_losses).mean()
        hessian_loss = torch.stack(batch_hessian_losses).mean()

        # Energy loss (batch)
        energy_loss = loss_fn_energy(predicted_energy, target_energy)

        total_batch_loss = ENERGY_WEIGHT * energy_loss + FORCE_WEIGHT * force_loss + HESSIAN_WEIGHT * hessian_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        with torch.no_grad():
            for name, param in student.named_parameters():
                if name in ema_weights:
                    ema_weights[name].mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

        total_loss += total_batch_loss.item()
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()
        total_hessian_loss += hessian_loss.item()

        loop.set_postfix({
            "Loss": total_batch_loss.item(),
            "E": energy_loss.item(),
            "F": force_loss.item(),
            "H": hessian_loss.item()
        })

    scheduler.step(total_loss / min_len)
    losses.append(total_loss / min_len)
    print(f"Epoch {epoch+1}: Loss={total_loss/min_len:.4f}, Energy={total_energy_loss/min_len:.4f}, Force={total_force_loss/min_len:.4f}, Hessian={total_hessian_loss/min_len:.4f}")

# Apply EMA weights before saving
for name, param in student.named_parameters():
    if name in ema_weights:
        param.data.copy_(ema_weights[name])

# Save model
model_path = os.path.join(RESULTS_DIR, "fine_tuned_student_with_hessian_permol_torchani.model")
torch.save(student, model_path)
print(f"✅ Training complete. Model saved to {model_path}")

# Plot losses
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(os.path.join(RESULTS_DIR, "loss_plot.png"))
plt.close()