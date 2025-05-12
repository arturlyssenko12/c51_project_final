import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read

# Load dataset
XYZ_PATH = "BOTNet-datasets/dataset_3BPA/test_dih.xyz"
atoms_list = read(XYZ_PATH, ":")

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get initial atomic positions from the first structure
positions = torch.tensor(atoms_list[0].get_positions(), dtype=dtype, device=device)
print(positions)
n_coords = positions.numel()

# Generate symmetric positive semi-definite matrix A for synthetic quadratic energy
# A = torch.randn(n_coords, n_coords, dtype=dtype, device=device)
A = torch.eye(n_coords, dtype=dtype, device=device)*.1
A = A @ A.T

def energy_fn(pos):
    flat = pos.view(-1)
    return 0.5 * flat @ A @ flat
print(energy_fn(positions))
# Ground truth Hessian via autograd
positions_autograd = positions.clone().detach().requires_grad_(True)
flat_pos = positions_autograd.view(-1)
energy = energy_fn(positions_autograd)
force = -torch.autograd.grad(energy, positions_autograd, create_graph=True)[0].view(-1)

true_rows = []
for i in range(min(10, n_coords)):
    row = torch.autograd.grad(force[i], positions_autograd, retain_graph=True)[0].view(-1)
    true_rows.append(row)
true_hessian = torch.stack(true_rows)

def finite_difference_hessian_rows(energy_fn, pos, indices, epsilon):
    pos = pos.detach()
    eye = torch.eye(pos.numel(), device=pos.device, dtype=pos.dtype)[indices]
    perturb = epsilon * eye.view(-1, *pos.shape)

    pos_plus = pos.unsqueeze(0) + perturb
    pos_minus = pos.unsqueeze(0) - perturb

    all_energies = torch.stack([energy_fn(p) for p in torch.cat([pos_plus, pos_minus], dim=0)])
    e_plus, e_minus = all_energies.chunk(2, dim=0)

    hess_rows = ((e_plus - e_minus) / (2 * epsilon)).view(len(indices), -1)
    return hess_rows

# Sweep epsilons and compute error
epsilons = [10**i for i in range(-5, 4)]
results = []
mse_loss_fn = torch.nn.MSELoss()

indices = list(range(min(10, n_coords)))

for eps in epsilons:
    start = time.time()
    approx_hess = finite_difference_hessian_rows(energy_fn, positions, indices, epsilon=eps)
    runtime = time.time() - start

    abs_error = mse_loss_fn(approx_hess, true_hessian).item()
    rel_error = abs_error / mse_loss_fn(true_hessian, torch.zeros_like(true_hessian)).item()
    results.append((eps, abs_error, rel_error, runtime))

# Save results
df = pd.DataFrame(results, columns=["epsilon", "absolute_error", "relative_error", "runtime_sec"])
print(df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 7))
plt.plot(df["epsilon"], df["absolute_error"], marker="o", label="Absolute Error (MSE)")
plt.plot(df["epsilon"], df["relative_error"], marker="s", label="Relative Error (MSE norm.)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Epsilon (log scale)")
plt.ylabel("Error")
plt.title("Hessian Approximation Error vs Epsilon (MSE-based)")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("hessian_error_vs_epsilon_mse.png")
