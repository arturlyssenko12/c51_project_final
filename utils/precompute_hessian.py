import os
import torch
from ase.io import read
from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
from tqdm import tqdm


def compute_full_hessian(forces, positions):
    """
    Compute the full Hessian matrix of the potential energy with respect to atomic positions.

    This function uses PyTorch autograd to compute the second derivatives of the energy 
    by differentiating the force vector (-∇E) with respect to positions. It performs a 
    vector-Jacobian product (VJP) for each degree of freedom to assemble the full Hessian.

    Args:
        forces (torch.Tensor): Tensor of shape (N, 3) representing the force on each atom.
                               Should be the negative gradient of the energy w.r.t. positions.
        positions (torch.Tensor): Tensor of shape (N, 3) representing atomic positions.
                                  Requires `requires_grad=True` for autograd to work.

    Returns:
        torch.Tensor: A tensor of shape (3N, 3N) representing the full Hessian matrix H,
                      where H[i, j] = ∂²E / ∂x_i ∂x_j. The indexing follows the flattened
                      positions: i = 3 * atom_index + coordinate (x/y/z).
    """
    n_atoms = positions.shape[0]
    n_dof = n_atoms * 3

    flat_forces = forces.view(-1)
    hessian_rows = []
    for i in range(n_dof):
        vec = torch.zeros_like(flat_forces)
        vec[i] = 1.0
        grad2 = torch.autograd.grad(
            flat_forces,
            positions,
            grad_outputs=vec.view_as(flat_forces),
            retain_graph=True
        )[0]
        hessian_rows.append(grad2.view(-1))

    hessian = torch.stack(hessian_rows, dim=0)
    return -hessian


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


def move_batch(batch, device, dtype=torch.float64):
    batch = batch.to(device)
    for k, v in batch.to_dict().items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            setattr(batch, k, v.to(dtype))
    return batch


def main():
    # Setup
    model_path = "/home/alyssenko/c51_project/MACE-OFF23_large.model"
    data_path = "/home/alyssenko/c51_project/BOTNet-datasets/dataset_3BPA/train_300K.xyz"
    save_path = "all_hessians.pt"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    dtype = next(model.parameters()).dtype

    # Load data
    atoms_list = read(data_path, ":")
    teacher_data = create_atomic_data_list(atoms_list, model, "teacher")
    teacher_loader = torch_geometric.dataloader.DataLoader(teacher_data, batch_size=1, shuffle=False)

    # Storage lists
    energies = []
    forces = []
    hessians = []

    # Process each structure
    for batch in tqdm(teacher_loader):
        batch = move_batch(batch, device, dtype)
        batch.positions.requires_grad_(True)

        teacher_out = model(batch.to_dict())
        energy = teacher_out["energy"]
        force = teacher_out["forces"]
        hessian = compute_full_hessian(force, batch.positions)

        energies.append(energy.detach().cpu())
        forces.append(force.detach().cpu())
        hessians.append(hessian.detach().cpu())

    # Save everything in one file
    torch.save({
        "energy": energies,
        "forces": forces,
        "hessian": hessians,
    }, save_path)


if __name__ == "__main__":
    main()
