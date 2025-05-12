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

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN.E2GNN import E2GNN
from graph_constructor import AtomsToGraphs

from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric

# ------------------- Config -------------------
TEACHER_PATH = "MACE-OFF23_large.model"
XYZ_PATH = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
BATCH_SIZE = 4
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ------------------- Data helpers -------------------
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

def create_e2gnn_data_list(atoms_list, description):
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=5,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=False,
    )
    data_list = []
    dummy_forces = torch.tensor([0.0, 0.0, 0.0])
    for atoms in atoms_list:
        data = a2g.convert(atoms)
        data.y = 0
        data.force = dummy_forces
        data_list.append(data)
    print(f"Created {len(data_list)} samples for {description}")
    return data_list

def collate_fn_e2gnn(batch):
    return Batch.from_data_list(batch)

# ------------------- Load Models -------------------
teacher = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.to(device)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False
teacher_dtype = next(teacher.parameters()).dtype

student = E2GNN(
    hidden_channels=512,
    num_layers=4,
    num_rbf=128,
    cutoff=6.0,
    max_neighbors=20,
    use_pbc=False,
    otf_graph=True
).to(device=device, dtype=torch.float).train()

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
student_data = create_e2gnn_data_list(atoms_list, "student")

teacher_loader = torch_geometric.dataloader.DataLoader(
    teacher_data, batch_size=BATCH_SIZE, shuffle=True
)
student_loader = DataLoader(
    student_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_e2gnn
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

        batch_teacher = move_batch(batch_teacher, device, dtype=teacher_dtype)
        batch_student = move_batch(batch_student, device, dtype=torch.float)

        for b in [batch_teacher, batch_student]:
            if b.edge_index is not None:
                b.edge_index = b.edge_index.to(torch.long)
            if b.batch is not None:
                b.batch = b.batch.to(torch.long)
            if hasattr(b, "head") and b.head is not None:
                b.head = b.head.to(torch.long)

        batch_teacher.positions.requires_grad_(False)
        batch_student.positions = batch_teacher.positions.detach().clone()

        teacher_out = teacher(batch_teacher.to_dict())
        target_energy = teacher_out["energy"].detach().to(dtype=torch.float)
        target_forces = teacher_out["forces"].detach().to(dtype=torch.float)

        predicted_energy, predicted_forces = student(batch_student)

        energy_loss = loss_fn_energy(predicted_energy, target_energy)
        force_loss = loss_fn_force(predicted_forces, target_forces)

        n_atoms = batch_teacher.positions.size(0)
        n_coords = 3 * n_atoms
        indices = random.sample(range(n_coords), min(HESSIAN_ROWS, n_coords))

        def teacher_force_fn(pos):
            batch_teacher.positions = pos
            return teacher(batch_teacher.to_dict())["forces"].detach()

        def student_force_fn(pos):
            batch_student.positions = pos
            return student(batch_student)[1].detach()

        teacher_hess = finite_difference_hessian_rows_batched(teacher_force_fn, batch_teacher.positions, indices)
        student_hess = finite_difference_hessian_rows_batched(student_force_fn, batch_student.positions, indices)

        hessian_loss = sum(
            nn.functional.mse_loss(s, t) for s, t in zip(student_hess, teacher_hess)
        ) / len(indices)

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
model_path = os.path.join(RESULTS_DIR, "fine_tuned_student_with_hessian.model")
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