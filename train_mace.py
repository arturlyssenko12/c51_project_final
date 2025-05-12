import torch
import torch.nn as nn
from ase.io import read
from tqdm import tqdm
from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric


TEACHER_PATH = "2023-12-10-mace-128-L0_energy_epoch-249.model"
STUDENT_PATH = "small_MACE.model"
DATA_PATH = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"

NUM_EPOCHS = 100           # target 100–150 for small student models
BATCH_SIZE = 4             # if student is GemNet-dT
LEARNING_RATE = 1e-3       # from Table 7
FORCE_WEIGHT = 100.0       # λF = 100 across all tasks

loss_fn_energy = nn.MSELoss() 
loss_fn_force = nn.L1Loss()   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
student = torch.load(STUDENT_PATH, map_location=device, weights_only=False)

teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

teacher_dtype = next(teacher.parameters()).dtype
student = student.to(device=device, dtype=teacher_dtype).train()

# optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=2e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=5,min_lr=1e-6)
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

teacher_loader = torch_geometric.dataloader.DataLoader(
    teacher_data, batch_size=BATCH_SIZE, shuffle=True
)
student_loader = torch_geometric.dataloader.DataLoader(
    student_data, batch_size=BATCH_SIZE, shuffle=True
)

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0

    loop = tqdm(zip(teacher_loader, student_loader),
                total=len(teacher_loader),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch_teacher, batch_student in loop:
        batch_teacher = batch_teacher.to(dtype=teacher_dtype, device=device)
        batch_student = batch_student.to(dtype=teacher_dtype, device=device)

        for b in [batch_teacher, batch_student]:
            b.edge_index = b.edge_index.to(torch.long)
            b.batch = b.batch.to(torch.long)
            if hasattr(b, "head"):
                b.head = b.head.to(torch.long)

        batch_teacher.positions.requires_grad_(True)
        teacher_out = teacher(batch_teacher.to_dict())
        target_energy = teacher_out["energy"].detach()
        target_forces = teacher_out["forces"].detach()

        batch_student.positions.requires_grad_(True)
        student_out = student(batch_student.to_dict())
        predicted_energy = student_out["energy"]
        predicted_forces = student_out["forces"]

        n_atoms = batch_student.positions.shape[0]
        energy_loss = loss_fn_energy(predicted_energy, target_energy)
        force_loss = loss_fn_force(predicted_forces, target_forces)

        total_batch_loss = energy_loss + FORCE_WEIGHT * force_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()

        loop.set_postfix({
            "loss": total_batch_loss.item(),
            "E_loss": energy_loss.item(),
            "F_loss": force_loss.item() if FORCE_WEIGHT > 0 else 0.0
        })

    avg_loss = total_loss / len(teacher_loader)
    scheduler.step(avg_loss)
    avg_energy = total_energy_loss / len(teacher_loader)
    avg_force = total_force_loss / len(teacher_loader) if FORCE_WEIGHT > 0 else 0.0
    print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} Energy={avg_energy:.6f} Force={avg_force:.6f}")

torch.save(student, "fine_tuned_student.model")
print("Training complete. Saved to fine_tuned_student.model")
