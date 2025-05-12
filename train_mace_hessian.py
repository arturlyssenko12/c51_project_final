import torch
import torch.nn as nn
from ase.io import read
from tqdm import tqdm
from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
import random

# --- File paths ---
TEACHER_PATH = "2023-12-10-mace-128-L0_energy_epoch-249.model"
STUDENT_PATH = "small_MACE.model"
DATA_PATH = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"

# --- Optimal hyperparameters from the paper ---
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
FORCE_WEIGHT = 100.0           # λF
HESSIAN_WEIGHT = 400.0         # λKD
HESSIAN_ROWS = 1               # s (subsampled Hessian rows)
WEIGHT_DECAY = 2e-6
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.8
LR_MIN = 1e-6

# --- Loss functions ---
loss_fn_energy = nn.MSELoss()
loss_fn_force = nn.L1Loss()

# --- Load models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
student = torch.load(STUDENT_PATH, map_location=device, weights_only=False)

teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

teacher_dtype = next(teacher.parameters()).dtype
student = student.to(device=device, dtype=teacher_dtype).train()

# --- Optimizer & Scheduler ---
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
                                                        patience=LR_SCHEDULER_PATIENCE, min_lr=LR_MIN)

# --- Load dataset ---
print("Loading ASE atoms from:", DATA_PATH)
atoms_list = read(DATA_PATH, ":")

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

teacher_data = create_atomic_data_list(atoms_list, teacher, "teacher")
student_data = create_atomic_data_list(atoms_list, student, "student")

teacher_loader = torch_geometric.dataloader.DataLoader(teacher_data, batch_size=BATCH_SIZE, shuffle=True)
student_loader = torch_geometric.dataloader.DataLoader(student_data, batch_size=BATCH_SIZE, shuffle=True)

def compute_hessian_rows(tensor, positions, select_indices):
    """Compute selected Hessian (second derivative) rows using VJP."""
    flat_tensor = tensor.view(-1)
    hessian_rows = []
    for idx in select_indices:
        one_hot = torch.zeros_like(flat_tensor)
        one_hot[idx] = 1.0
        grad_outputs = one_hot.view_as(tensor)
        grads = torch.autograd.grad(
            outputs=tensor,
            inputs=positions,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        hessian_rows.append(grads.view(-1))
    return hessian_rows

# --- Training loop ---
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_hessian_loss = 0.0

    loop = tqdm(zip(teacher_loader, student_loader),
                total=len(teacher_loader),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch_teacher, batch_student in loop:
        # Move to device
        batch_teacher = batch_teacher.to(dtype=teacher_dtype, device=device)
        batch_student = batch_student.to(dtype=teacher_dtype, device=device)

        for b in [batch_teacher, batch_student]:
            b.edge_index = b.edge_index.to(torch.long)
            b.batch = b.batch.to(torch.long)
            if hasattr(b, "head"):
                b.head = b.head.to(torch.long)

        batch_teacher.positions.requires_grad_(True)
        batch_student.positions.requires_grad_(True)

        # --- Teacher forward ---
        teacher_out = teacher(batch_teacher.to_dict())
        target_energy = teacher_out["energy"].detach()
        target_forces = teacher_out["forces"].detach()

        # --- Student forward ---
        student_out = student(batch_student.to_dict())
        predicted_energy = student_out["energy"]
        predicted_forces = student_out["forces"]

        # --- Losses ---
        energy_loss = loss_fn_energy(predicted_energy, target_energy)
        force_loss = loss_fn_force(predicted_forces, target_forces)

        # --- Hessian KD Loss ---
        total_atoms = batch_student.positions.shape[0]
        total_coords = total_atoms * 3
        selected_indices = random.sample(range(total_coords), min(HESSIAN_ROWS, total_coords))

        teacher_hessian_rows = compute_hessian_rows(
            teacher_out["forces"], batch_teacher.positions, selected_indices
        )
        student_hessian_rows = compute_hessian_rows(
            student_out["forces"], batch_student.positions, selected_indices
        )

        hessian_loss = sum(
            nn.functional.mse_loss(s_row, t_row.detach())
            for s_row, t_row in zip(student_hessian_rows, teacher_hessian_rows)
        ) / len(selected_indices)

        total_batch_loss = (
            energy_loss +
            FORCE_WEIGHT * force_loss +
            HESSIAN_WEIGHT * hessian_loss
        )

        # --- Optimization Step ---
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # --- Logging ---
        total_loss += total_batch_loss.item()
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()
        total_hessian_loss += hessian_loss.item()

        loop.set_postfix({
            "loss": total_batch_loss.item(),
            "E_loss": energy_loss.item(),
            "F_loss": force_loss.item(),
            "H_loss": hessian_loss.item()
        })

    avg_loss = total_loss / len(teacher_loader)
    avg_energy = total_energy_loss / len(teacher_loader)
    avg_force = total_force_loss / len(teacher_loader)
    avg_hessian = total_hessian_loss / len(teacher_loader)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} | Energy={avg_energy:.6f} | Force={avg_force:.6f} | Hessian={avg_hessian:.6f}")

# --- Save final model ---
torch.save(student, "fine_tuned_student_HESSIAN.model")
print("Training complete. Saved to fine_tuned_student.model")