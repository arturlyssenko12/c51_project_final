import os
import torch
from ase.io import read
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
#from mace.data import config_from_atoms, AtomicData, KeySpecification
#from mace.tools.utils import AtomicNumberTable
#from mace.tools import torch_geometric
from tqdm import tqdm

import torchani
from torchani import AEVComputer
from torchani.data import collate_fn as collate_fn_torchani
from torchani.utils import ChemicalSymbolsToInts
# --------------- Torchani layers settings ---------------

torchspecies = torchani.models.ANI1x().species
species_converter = ChemicalSymbolsToInts(torchspecies)

def build_nn():
    
    H_network = torch.nn.Sequential(
        torch.nn.Linear(384, 160),
        torch.nn.CELU(0.1),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    C_network = torch.nn.Sequential(
        torch.nn.Linear(384, 144),
        torch.nn.CELU(0.1),
        torch.nn.Linear(144, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    N_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    O_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )
    
    nets = [H_network, C_network, N_network, O_network]

    for net in nets:
        net.apply(init_normal)
    return nets

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)

def collate_fn(batch):
    species, coords, energy, forces, hessian = zip(*batch)
    return {
        'species': torch.stack(species),
        'coordinates': torch.stack(coords),
        'energy': torch.stack(energy),
        'forces': torch.stack(forces),
        'hessian': torch.stack(hessian)
    }

class XYZ: 
    def __init__(self, filename, device):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # parse lines

        self.mols = []
        atom_count = None
        species = []
        coordinates = []
        state = 'ready'
        for i in lines:
            i = i.strip()
            if state == 'ready':
                atom_count = int(i)
                state = 'comment'
            elif state == 'comment':
                state = 'atoms'
            else:
                parts = i.split()
                s = parts[0]
                x, y, z = map(float, parts[1:4])
                species.append(s)
                coordinates.append([x, y, z])
                atom_count -= 1
                if atom_count == 0:
                    state = 'ready'
                    species = species_converter(species) \
                        .to(device)
                    coordinates = torch.tensor(coordinates, device=device)
                    self.mols.append((species, coordinates))
                    coordinates = []
                    species = []

    def _len_(self):
        return len(self.mols)

    def _getitem_(self, i):
        species, coords = self.mols[i]
        return {'species': species, 'coordinates': coords}
    
class AtomsDataset(Dataset):
    def __init__(self,
                 species,
                 coords,
                 target_dict,
                 device,
                 energy_mean: torch.Tensor = None,
                 energy_std:  torch.Tensor = None,
                 hessian_scale: torch.Tensor = None,
                 plot_hist: bool = False):
        self.species = species
        self.coords = coords
        self.device = device
        #self.a2g        = AtomsToGraphs(
        #    max_neigh=50, radius=5,
        #    r_energy=False, r_forces=False,
        #    r_fixed=True, r_distances=False,
        #    r_edges=False
        #)

        #all_e = torch.stack([e.flatten()[0] for e in target_dict["energy"]]).to(torch.float32)
        #print(f"all_e shape: {all_e.shape}")

        #for k, v in target_dict.items():
        #    print(f"{k}: {type(v)}, length={len(v)}")
        #   if len(v) > 0:
        #       print(f"  First item: type={type(v[0])}, shape={v[0].shape}")

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
            e = e.clone().detach().to(dtype=torch.float)    
            f = f.clone().detach().to(dtype=torch.float)    
            h = h.clone().detach().to(dtype=torch.float)    

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
        return len(self.species)

    def __getitem__(self, idx):
        species = self.species[idx]
        coords = self.coords[idx]
        tgt = self.target_data[idx]
        return (
            species.to(self.device), 
            coords.to(self.device), 
            tgt["energy"].to(self.device), 
            tgt["forces"].to(self.device), 
            tgt["hessian"].to(self.device)
        )
    
