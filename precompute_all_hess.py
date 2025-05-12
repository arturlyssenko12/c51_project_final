import os
import torch
from tqdm import tqdm
from ase.io import read
from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric


def compute_full_hessian(forces, positions):
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


def process_xyz_file(xyz_path, model, device):
    atoms_list = read(xyz_path, ":")
    teacher_data = create_atomic_data_list(atoms_list, model, f"{xyz_path}")
    loader = torch_geometric.dataloader.DataLoader(teacher_data, batch_size=1, shuffle=False)

    energies = []
    forces = []
    hessians = []

    for batch in tqdm(loader, desc=os.path.basename(xyz_path)):
        batch = move_batch(batch, device)
        batch.positions.requires_grad_(True)
        teacher_out = model(batch.to_dict())
        energy = teacher_out["energy"]
        force = teacher_out["forces"]
        hessian = compute_full_hessian(force, batch.positions)

        energies.append(energy.detach().cpu())
        forces.append(force.detach().cpu())
        hessians.append(hessian.detach().cpu())

    save_path = os.path.join(os.path.dirname(xyz_path), f"precomputed_training_data_{os.path.basename(xyz_path).replace('.xyz', '')}.pt")
    torch.save({
        "energy": energies,
        "forces": forces,
        "hessian": hessians,
    }, save_path)


def main():
    root_dir = "BOTNet-datasets"
    model_path = "MACE-OFF23_large.model"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:1"

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_path) or not dataset_name.startswith("dataset_"):
            continue
        for file in os.listdir(dataset_path):
            if file.endswith(".xyz"):
                xyz_file = os.path.join(dataset_path, file)
                try:
                    process_xyz_file(xyz_file, model, device)
                except Exception as e:
                    print(f"Failed to process {xyz_file}: {e}")


if __name__ == "__main__":
    main()