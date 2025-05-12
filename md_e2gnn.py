#!/usr/bin/env python3

import os
import sys
import time
import csv
from pathlib import Path

import torch
import numpy as np

from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list

from torch_geometric.data import Data, Batch
from loaders.loaders import AtomsDataset, move_batch

from tqdm import tqdm

# Add your E2GNN folder to PATH and import the model class
sys.path.append(os.path.join(os.getcwd(), "E2GNN"))
from E2GNN import E2GNN


def pack_molecules(
    xyz_path: str,
    N: int,
    cell_size: float,
    min_dist: float,
    rng_seed: int = 42
) -> Atoms:
    """Pack N non‐overlapping copies of the first frame of an XYZ into a box."""
    mol = read(xyz_path, index=0)
    base_pos = mol.get_positions()
    symbols = mol.get_chemical_symbols()
    nat0 = len(mol)

    from collections import defaultdict
    grid = defaultdict(list)
    cell_side = min_dist

    def cell_idx(pt):
        return tuple((pt // cell_side).astype(int))

    nbrs = [(i, j, k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)]
    rng = np.random.default_rng(rng_seed)

    packed_pos = []
    packed_sym = []
    placed = 0
    attempts = 0
    max_attempts = N * 10000

    while placed < N:
        shift = rng.random(3) * cell_size
        new_pos = base_pos + shift
        ok = True
        for pt in new_pos:
            ci = cell_idx(pt)
            for di, dj, dk in nbrs:
                for prev in grid[(ci[0]+di, ci[1]+dj, ci[2]+dk)]:
                    if np.sum((pt - prev)**2) < min_dist**2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to place {N} copies clash‐free")
            continue
        for pt in new_pos:
            grid[cell_idx(pt)].append(pt)
        packed_pos.extend(new_pos.tolist())
        packed_sym.extend(symbols)
        placed += 1

    return Atoms(
        symbols=packed_sym,
        positions=np.array(packed_pos),
        cell=[cell_size]*3,
        pbc=True
    )


def combine_xyz_files(paths):
    frames = []
    for p in paths:
        frames.extend(read(p, ":"))
    return frames


def combine_target_dicts(paths):
    tgt = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        tgt["energy"].extend(d["energy"])
        tgt["forces"].extend(d["forces"])
        tgt["hessian"].extend(d["hessian"])
    return tgt


def atoms_to_batch(atoms: Atoms) -> Data:
    """Convert ASE Atoms → torch_geometric Data (positions, Z, cell, natoms)."""
    pos = torch.tensor(atoms.get_positions(),      dtype=torch.float32)
    z   = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    cell= torch.tensor(atoms.get_cell(),           dtype=torch.float32).view(1,3,3)
    nat = atoms.get_number_of_atoms()
    return Data(pos=pos, atomic_numbers=z, cell=cell, natoms=torch.tensor([nat]))


class Normalizer:
    """Simple E‐mean/std ↔ normalized‐E converter."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std  = std
    def denorm(self, e_norm: torch.Tensor) -> torch.Tensor:
        return e_norm * self.std + self.mean


class E2GNNCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, model, normalizer: Normalizer, cutoff: float, device: str):
        super().__init__()
        self.device     = device
        self.model      = model.to(device).eval()
        self.normalizer = normalizer
        self.cutoff     = cutoff

    def calculate(self, atoms, properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 1) on‐the‐fly neighbor list
        i_idx, j_idx = neighbor_list(
            "ij", atoms,
            cutoff=self.cutoff,
            self_interaction=False,
        )
        i_t = torch.tensor(i_idx, dtype=torch.long, device=self.device)
        j_t = torch.tensor(j_idx, dtype=torch.long, device=self.device)

        # 2) build graph Data
        data = atoms_to_batch(atoms)
        data.pos            = data.pos.to(self.device)
        data.atomic_numbers = data.atomic_numbers.to(self.device)
        data.cell           = data.cell.to(self.device)
        data.edge_index     = torch.stack([i_t, j_t], dim=0)
        vec = data.pos[j_t] - data.pos[i_t]
        data.edge_weight    = vec.norm(dim=1, keepdim=True)

        batch = Batch.from_data_list([data]).to(self.device)
        batch = move_batch(batch, self.device, torch.float)

        # 3) inference
        with torch.no_grad():
            e_norm, f_norm = self.model(batch)
            e = self.normalizer.denorm(e_norm)

        # 4) record
        energy = float(e.item())
        forces = f_norm.cpu().numpy() * self.normalizer.std
        self.results["energy"] = energy
        self.results["forces"] = np.ascontiguousarray(forces, dtype=float)


def run_md(
    atoms: Atoms,
    model_path: str,
    output_dir: str,
    dt_fs: float = 1.0,
    n_steps: int = 1000,
    temperature_K: float = 300.0,
    friction: float = 0.01,
    log_interval: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    natoms = atoms.get_number_of_atoms()

    # normalization from training set
    XYZ_TRAIN = [
        "BOTNet-datasets/dataset_3BPA/train_300K.xyz",
        "BOTNet-datasets/dataset_3BPA/train_mixedT.xyz",
        "BOTNet-datasets/dataset_3BPA/test_dih.xyz",
    ]
    PT_TRAIN = [
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_300K.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_mixedT.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_dih.pt",
    ]
    train_atoms = combine_xyz_files(XYZ_TRAIN)
    train_tgt   = combine_target_dicts(PT_TRAIN)
    all_h = torch.cat([h.flatten() for h in train_tgt["hessian"]])
    train_ds = AtomsDataset(
        atoms_list=train_atoms,
        target_dict=train_tgt,
        device=device,
        energy_mean=None,
        energy_std=None,
        hessian_scale=all_h.std().item(),
        plot_hist=False,
    )
    normalizer = Normalizer(
        mean=float(train_ds.energy_mean),
        std = float(train_ds.energy_std),
    )
    CUTOFF = 4.5
    # load E2GNN model
    model = E2GNN(
        hidden_channels=128,
        num_layers=3,
        num_rbf=32,
        cutoff=CUTOFF,
        max_neighbors=15,
        use_pbc=False,
        otf_graph=False,
        num_elements=9,
    )
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device).eval()
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    # attach calculator
    atoms.calc = E2GNNCalculator(model, normalizer, cutoff=CUTOFF, device=device)

    # init velocities & zero total momentum
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    mom = atoms.get_momenta()
    atoms.set_momenta(mom - mom.mean(axis=0))

    # integrator
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )

    # trajectory writer
    traj = Trajectory(os.path.join(output_dir, "traj.traj"), 'w', atoms)
    dyn.attach(traj.write, interval=log_interval)

    # ASE MDLogger
    logger = MDLogger(
        dyn, atoms,
        logfile=sys.stdout,
        header=True,
        stress=False,
        peratom=False,
        mode="a",
    )
    dyn.attach(logger, interval=log_interval)

    # CSV logger with ns/day
    csv_path = os.path.join(output_dir, "thermo.csv")
    fcsv = open(csv_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow([
        "ps", "Etot/N", "Epot/N", "Ekin/N", "T[K]",
        "inst_ns_per_day", "avg_ns_per_day",
    ])
    wall = {"last": time.perf_counter(), "cum_wall": 0.0, "cum_sim": 0.0}

    def write_csv(a=atoms):
        t = dyn.get_time() / (1000 * units.fs)
        ep = a.get_potential_energy() / natoms
        ek = a.get_kinetic_energy()  / natoms
        tp = a.get_temperature()

        now   = time.perf_counter()
        delta = now - wall["last"]
        wall["last"] = now
        wall["cum_wall"] += delta
        wall["cum_sim"]  += dt_fs * log_interval

        dt_ns = dt_fs * log_interval * 1e-6
        inst  = dt_ns / delta * 86400.0
        avg   = (wall["cum_sim"]*1e-6) / wall["cum_wall"] * 86400.0

        writer.writerow([
            f"{t:.4f}", f"{ep:.4f}", f"{ek:.4f}", f"{tp:.1f}",
            f"{inst:.2f}", f"{avg:.2f}"
        ])

    dyn.attach(write_csv, interval=log_interval)

    # # progress bar
    # pbar = tqdm(total=n_steps, desc="MD")
    # dyn.attach(lambda: pbar.update(log_interval), interval=log_interval)

    # run!
    dyn.run(n_steps)

    fcsv.close()
    traj.close()
    # pbar.close()
    print(f"MD finished — outputs saved to {output_dir}")


if __name__ == "__main__":
    XYZ      = "BOTNet-datasets/dataset_3BPA/test_300K.xyz"
    MODEL    = "e2gnn_student_supervised_HESSIAN.model"


    N_arr = np.logspace(0,10,base=2,num=10).astype(int)
    for N in  N_arr:
        print(f"--- Running MD with N={N} copies ---")
        atoms = pack_molecules(
            xyz_path=XYZ,
            N=N,
            cell_size=100.0,
            min_dist=2.5,
            rng_seed=123,
        )
        out_dir = os.path.join(f"results/e2gnn_hessian", str(N))
        run_md(
            atoms=atoms,
            model_path=MODEL,
            output_dir=out_dir,
            dt_fs=1.,
            n_steps=100,
            temperature_K=300.0,
            friction=0.01,
            log_interval=10,
        )
