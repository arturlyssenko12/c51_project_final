import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io import read
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from loaders.loaders import AtomsDataset, move_batch,collate_fn_e2gnn
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN


# ------------------- Config -------------------
XYZ_PATH_TRAIN = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
XYZ_PATH_TEST = "BOTNet-datasets/dataset_3BPA/test_300K.xyz"
TARGET_PATH = "/home/alyssenko/c51_project/utils/all_hessians.pt"
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
FORCE_WEIGHT = 100.0
# device = "cpu"
device = "cuda:0"

# ------------------- Dataset -------------------



# === Data Loaders ===
target_dat = torch.load(TARGET_PATH)
atoms_list = read(XYZ_PATH_TRAIN, ":")
train_dataset = AtomsDataset(atoms_list, target_dat, device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_e2gnn)

# === Model ===
# student = E2GNN(
#     hidden_channels=128,
#     num_layers=4,
#     num_rbf=32,
#     cutoff=6.0,
#     max_neighbors=20,
#     use_pbc=False,
#     otf_graph=True,
#     num_elements=max(a.number for atoms in atoms_list for a in atoms) + 1
# ).to(device).to(dtype=torch.float).train()


student = E2GNN(
    hidden_channels=128,
    num_layers=3,
    num_rbf=32,
    cutoff=4.5,        
    max_neighbors=15,  
    use_pbc=False,
    otf_graph=True,
    num_elements=9
).to(device).to(dtype=torch.float).train()


optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=2e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-6)
loss_fn_energy = nn.MSELoss()
loss_fn_force = nn.MSELoss()

# === Training ===
for epoch in range(NUM_EPOCHS):
    total_loss = total_energy_loss = total_force_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in loop:
        batch = move_batch(batch, device, torch.float)
        pred_energy, pred_forces = student(batch)
        energy_loss = loss_fn_energy(pred_energy, batch.y)
        force_loss = loss_fn_force(pred_forces, batch.force)
        loss = energy_loss + FORCE_WEIGHT * force_loss
        import pdb
        pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()
        loop.set_postfix(Loss=loss.item(), E=energy_loss.item(), F=force_loss.item())

    scheduler.step(total_loss / len(train_loader))
    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
          f"Energy={total_energy_loss/len(train_loader):.4f}, "
          f"Force={total_force_loss/len(train_loader):.4f}")

torch.save(student, "fine_tuned_student_in_memory.model")
print("✅ Model saved to fine_tuned_student_in_memory.model")

# === Evaluation ===
student.eval()
atoms_list_test = read(XYZ_PATH_TEST, ":")
n_test = min(
    len(atoms_list_test),
    len(target_dat["energy"]),
    len(target_dat["forces"]),
    len(target_dat["hessian"]),
)
target_test = {
    "energy": target_dat["energy"][:n_test],
    "forces": target_dat["forces"][:n_test],
    "hessian": target_dat["hessian"][:n_test],
}
test_dataset = AtomsDataset(atoms_list_test[:n_test], target_test, device)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_e2gnn)

true_e, pred_e, true_f, pred_f = [], [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = move_batch(batch, device)
        e_pred, f_pred = student(batch)

        true_e.append(batch.y.cpu())
        pred_e.append(e_pred.cpu())
        true_f.append(batch.force.cpu())
        pred_f.append(f_pred.cpu())

true_e = torch.cat(true_e).squeeze().numpy()
pred_e = torch.cat(pred_e).squeeze().numpy()
true_f = torch.cat(true_f).view(-1, 3).numpy()
pred_f = torch.cat(pred_f).view(-1, 3).numpy()

# === Plots ===
plt.figure(figsize=(6, 6))
plt.scatter(true_e, pred_e, alpha=0.6, s=10)
plt.plot([true_e.min(), true_e.max()], [true_e.min(), true_e.max()], 'r--')
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("Energy Correlation")
plt.tight_layout()
plt.savefig("correlation_plot_energy.png")

plt.figure(figsize=(6, 6))
plt.scatter(true_f.flatten(), pred_f.flatten(), alpha=0.4, s=5)
plt.plot([true_f.min(), true_f.max()], [true_f.min(), true_f.max()], 'r--')
plt.xlabel("True Force Component")
plt.ylabel("Predicted Force Component")
plt.title("Force Correlation")
plt.tight_layout()
plt.savefig("correlation_plot_force.png")

print("✅ Correlation plots saved.")