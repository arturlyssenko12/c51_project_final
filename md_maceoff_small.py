#!/usr/bin/env python3

import os
import sys
import time
import csv

import torch
import numpy as np

from pathlib import Path
from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Import DataLoader for batching
from mace.tools import torch_geometric
from mace.data import config_from_atoms, AtomicData
from mace.tools.utils import AtomicNumberTable


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


class MACECustomCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, z_table, cutoff: float, device: str):
        super().__init__()
        self.device  = device
        # ensure model parameters & buffers are float32 on correct device
        self.model   = model.float().to(device).eval()
        self.z_table = z_table
        self.cutoff  = cutoff
        self.head_name = getattr(model, "available_heads", ["energy"])[0]

    def calculate(self, atoms, properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 1) build default config (pos, z, neighbors, etc.)
        config = config_from_atoms(atoms, head_name=self.head_name)

        # 2) package into AtomicData
        data = AtomicData.from_config(
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=[self.head_name],
        )

        # 3) wrap into a DataLoader to create a Batch with 'batch' key
        loader = torch_geometric.dataloader.DataLoader([data], batch_size=1)
        batch = next(iter(loader)).to(self.device)

        # cast graph indices to long
        batch.edge_index = batch.edge_index.to(torch.long)
        batch.batch      = batch.batch.to(torch.long)
        if hasattr(batch, "head"):
            batch.head = batch.head.to(torch.long)

        # 4) inference
        # with torch.no_grad():
        out = self.model(batch.to_dict(), compute_force=True)
        energy = out["energy"].item()
        forces = out["forces"].cpu().detach().numpy()

        # 5) record results
        natoms = atoms.get_number_of_atoms()
        self.results["energy"] = float(energy)
        self.results["forces"] = np.ascontiguousarray(forces.reshape(natoms, 3), dtype=float)


def run_md(
    atoms,
    model,
    z_table,
    cutoff: float,
    output_dir: str,
    dt_fs: float = 1.0,
    n_steps: int = 1000,
    temperature_K: float = 300.0,
    friction: float = 0.01,
    log_interval: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    atoms.calc = MACECustomCalculator(model, z_table, cutoff=cutoff, device=str(device))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    mom = atoms.get_momenta()
    atoms.set_momenta(mom - mom.mean(axis=0))

    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )

    traj = Trajectory(os.path.join(output_dir, "traj.traj"), 'w', atoms)
    dyn.attach(traj.write, interval=log_interval)

    logger = MDLogger(dyn, atoms, logfile=sys.stdout,
                      header=True, stress=False, peratom=False, mode="a")
    dyn.attach(logger, interval=log_interval)

    csv_path = os.path.join(output_dir, "thermo.csv")
    fcsv = open(csv_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow([
        "ps", "Etot/N", "Epot/N", "Ekin/N", "T[K]",
        "inst_ns_per_day", "avg_ns_per_day",
    ])
    wall = {"last": time.perf_counter(), "cum_wall": 0.0, "cum_sim": 0.0}

    def write_csv():
        t   = dyn.get_time() / (1000 * units.fs)
        ep  = atoms.get_potential_energy() / atoms.get_number_of_atoms()
        ek  = atoms.get_kinetic_energy()     / atoms.get_number_of_atoms()
        tp  = atoms.get_temperature()

        now   = time.perf_counter()
        delta = now - wall["last"]
        wall["last"]   = now
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
    dyn.run(n_steps)

    fcsv.close()
    traj.close()
    print(f"MD finished — outputs saved to {output_dir}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    STUDENT_PATH = "MACE-OFF23_small.model"
    print("Loading MACE student from:", STUDENT_PATH)
    student = torch.load(STUDENT_PATH, map_location=device, weights_only=False)
    # convert student to float32
    student = student.float().to(device).eval()

    z_table = AtomicNumberTable([int(z) for z in student.atomic_numbers])
    cutoff  = 4.5

    XYZ = "BOTNet-datasets/dataset_3BPA/test_300K.xyz"
    N_arr = np.logspace(0,10,base=2,num=10).astype(int)
    for N in N_arr:
        print(f"--- Running MD with N={N} copies ---")
        atoms = pack_molecules(
            xyz_path=XYZ,
            N=N,
            cell_size=100.0,
            min_dist=2.5,
            rng_seed=123,
        )
        out_dir = os.path.join(f"results/mace_small", str(N))
        run_md(
            atoms=atoms,
            model=student,
            z_table=z_table,
            cutoff=cutoff,
            output_dir=out_dir,
            dt_fs=1.0,
            n_steps=1000,
            temperature_K=300.0,
            friction=0.01,
            log_interval=10,
        )
