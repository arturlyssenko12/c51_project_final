import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from ase.io import read
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


# --- Fix module import without changing E2GNN.py ---
e2gnn_dir = os.path.join(os.getcwd(), "E2GNN")
if e2gnn_dir not in sys.path:
    sys.path.insert(0, e2gnn_dir)

from E2GNN import E2GNN
from graph_constructor import AtomsToGraphs
from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric


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

def create_e2gnn_data_list(atoms_list, target_dict, description, device):
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

    if isinstance(target_dict, dict):
        target_data_list = [
            {
                "energy": target_dict["energy"][i],
                "forces": target_dict["forces"][i],
                "hessian": target_dict["hessian"][i],
            }
            for i in range(len(target_dict["energy"]))
        ]
    elif isinstance(target_dict, list):
        target_data_list = target_dict
    else:
        raise TypeError("target_dict must be either a dict of lists or a list of dicts.")

    for atoms, target_dat in zip(atoms_list, target_data_list):
        data = a2g.convert(atoms)
        energy = target_dat["energy"]
        forces = target_dat["forces"]
        hessian = target_dat["hessian"]

        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, dtype=torch.float64)
        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces, dtype=torch.float64)
        if not isinstance(hessian, torch.Tensor):
            hessian = torch.tensor(hessian, dtype=torch.float64)

        data.y = energy.view(1).to(device=device, dtype=torch.float32)
        data.force = forces.to(device=device, dtype=torch.float32)
        data.hessian = hessian.to(device=device, dtype=torch.float32)

        # # Add atomic_numbers field
        # atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        # data.atomic_numbers = atomic_numbers.to(device)

        data_list.append(data)

    print(f"Created {len(data_list)} samples for {description}")
    return data_list




def move_batch(batch, device, dtype=torch.float64):
    batch = batch.to(device)
    for k, v in batch.to_dict().items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            setattr(batch, k, v.to(dtype))
    return batch



# class AtomsDataset(torch.utils.data.Dataset):
#     def __init__(self, atoms_list, target_dict, device):
#         self.atoms_list = atoms_list
#         self.target_data = [
#             {"energy": e, "forces": f, "hessian": h}
#             for e, f, h in zip(target_dict["energy"], target_dict["forces"], target_dict["hessian"])
#         ]
#         self.device = device
#         self.a2g = AtomsToGraphs(max_neigh=50, radius=5, r_energy=False, r_forces=False,
#                                  r_fixed=True, r_distances=False, r_edges=False)

#     def __len__(self):
#         return len(self.atoms_list)

#     def __getitem__(self, idx):
#         atoms = self.atoms_list[idx]
#         data = self.a2g.convert(atoms)
#         targets = self.target_data[idx]

#         data.y = targets["energy"].view(1).to(self.device, dtype=torch.float32)
#         data.force = targets["forces"].to(self.device, dtype=torch.float32)
#         data.hessian = targets["hessian"].to(self.device, dtype=torch.float32)
#         data.atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

#         return data

class AtomsDataset(Dataset):
    def __init__(self,
                 atoms_list,
                 target_dict,
                 device,
                 energy_mean: torch.Tensor = None,
                 energy_std:  torch.Tensor = None,
                 hessian_scale: torch.Tensor = None,
                 plot_hist: bool = False):
        self.atoms_list = atoms_list
        self.device     = device
        self.a2g        = AtomsToGraphs(
            max_neigh=50, radius=5,
            r_energy=False, r_forces=False,
            r_fixed=True, r_distances=False,
            r_edges=False
        )

        # Energy statistics
        all_e = torch.tensor(target_dict["energy"], dtype=torch.float32)
        if energy_mean is None or energy_std is None:
            self.energy_mean = all_e.mean()
            self.energy_std  = all_e.std()
        else:
            self.energy_mean = float(energy_mean)
            self.energy_std  = float(energy_std)

        # Hessian normalization scale
        self.hessian_scale = (
            self.energy_std if hessian_scale is None else hessian_scale
        )

        # Normalize and store targets
        self.target_data = []
        all_f, all_h = [], []
        for e, f, h in zip(target_dict["energy"],
                           target_dict["forces"],
                           target_dict["hessian"]):
            # CPU tensor: clone and detach
            e = e.clone().detach().to(dtype=torch.float32)    
            f = f.clone().detach().to(dtype=torch.float32)    
            h = h.clone().detach().to(dtype=torch.float32)    

            e_norm = (e - self.energy_mean) / self.energy_std
            f_norm = f                / self.energy_std
            h_norm = h                / self.hessian_scale

            self.target_data.append({
                "energy":  e_norm,
                "forces":  f_norm,
                "hessian": h_norm
            })
            if plot_hist:
                all_f.append(f_norm.view(-1))
                all_h.append(h_norm.view(-1))

        if plot_hist:
            self._plot_histograms(all_e, all_f, all_h)

    def _plot_histograms(self, all_energies, all_f, all_h):
        all_energies = all_energies.cpu().numpy()
        all_f = torch.cat(all_f).cpu().numpy()
        all_h = torch.cat(all_h).cpu().numpy()

        plt.figure(); plt.hist(all_energies, bins=100); plt.title("Energy"); plt.savefig("energy.png"); plt.close()
        plt.figure(); plt.hist((all_energies - float(self.energy_mean.cpu()))/float(self.energy_std.cpu()), bins=100)
        plt.title("Energy Norm"); plt.savefig("energy_norm.png"); plt.close()
        plt.figure(); plt.hist(all_f, bins=100); plt.title("Force Norm"); plt.savefig("force_norm.png"); plt.close()
        plt.figure(); plt.hist(all_h, bins=100); plt.title("Hessian Norm"); plt.savefig("hessian_norm.png"); plt.close()

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        data  = self.a2g.convert(atoms)
        tgt   = self.target_data[idx]

        # Energy as a 1-element tensor
        data.y = torch.tensor([tgt["energy"].item()], device=self.device)
        data.force   = tgt["forces"].to(device=self.device)
        data.hessian = tgt["hessian"].to(device=self.device)
        data.atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        return data


def collate_fn_e2gnn(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for data in data_list:
                if data.edge_index is not None:
                    n_neighbors.append(data.edge_index.shape[1])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError, AttributeError):
            logging.warning("Missing edge_index info. Set otf_graph=True if computing edges on the fly.")

    return batch





