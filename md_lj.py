#!/usr/bin/env python3
"""
Run molecular dynamics using a simple Lennard-Jones 12-6 force field for 3BPA packed systems.
Logs thermodynamics and writes trajectories for multiple pack sizes.
"""
import argparse
import os,sys
import time
import csv
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.md import MDLogger


def pack_molecules(
    xyz_path: str,
    N: int,
    cell_size: float,
    min_dist: float,
    rng_seed: int = 42
) -> Atoms:
    """Pack N non-overlapping copies of the first frame of an XYZ into a box."""
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
                raise RuntimeError(f"Failed to place {N} copies clash-free")
            continue
        for pt in new_pos:
            grid[cell_idx(pt)].append(pt)
        packed_pos.extend(new_pos.tolist())
        packed_sym.extend(symbols)
        placed += 1

    return Atoms(
        symbols=packed_sym,
        positions=np.array(packed_pos),
        cell=[cell_size] * 3,
        pbc=True
    )


class LJCalculator(Calculator):
    """Simple Lennard-Jones 12-6 calculator with Lorentz-Berthelot mixing."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, sigma_map, epsilon_map, cutoff: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma_map = sigma_map
        self.epsilon_map = epsilon_map
        self.cutoff = cutoff

    def calculate(self, atoms, properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions(wrap=True)
        Zs = atoms.get_atomic_numbers()
        N = len(atoms)

        i_idx, j_idx = neighbor_list(
            "ij", atoms,
            cutoff=self.cutoff,
            self_interaction=False
        )

        energy = 0.0
        forces = np.zeros((N, 3), dtype=float)

        for i, j in zip(i_idx, j_idx):
            zi, zj = Zs[i], Zs[j]
            ri, rj = positions[i], positions[j]
            r_vec = ri - rj
            r = np.linalg.norm(r_vec)
            if r > self.cutoff or r < 1e-12:
                continue

            sigma_ij = 0.5 * (self.sigma_map[zi] + self.sigma_map[zj])
            epsilon_ij = np.sqrt(
                self.epsilon_map[zi] * self.epsilon_map[zj]
            )

            sr6 = (sigma_ij / r) ** 6
            sr12 = sr6 * sr6
            energy += 4.0 * epsilon_ij * (sr12 - sr6)

            f_mag = 24.0 * epsilon_ij * (2.0 * sr12 - sr6) / r
            f_vec = f_mag * (r_vec / r)
            forces[i] += f_vec
            forces[j] -= f_vec

        self.results["energy"] = energy
        self.results["forces"] = forces


def run_md(
    atoms: Atoms,
    sigma_map, epsilon_map,
    cutoff: float,
    output_dir: str,
    dt_fs: float,
    n_steps: int,
    temperature_K: float,
    friction: float,
    log_interval: int
):
    os.makedirs(output_dir, exist_ok=True)

    # attach LJ calculator
    atoms.calc = LJCalculator(sigma_map, epsilon_map, cutoff)

    # initialize velocities
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

    # log thermo to stdout
    logger = MDLogger(
        dyn, atoms,
        logfile=sys.stdout,
        header=True,
        stress=False,
        peratom=False,
        mode="a",
    )
    dyn.attach(logger, interval=log_interval)

    # csv logging
    csv_path = os.path.join(output_dir, "thermo.csv")
    fcsv = open(csv_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow([
        "ps", "Etot", "Epot", "Ekin", "T[K]",
        "inst_ns_per_day", "avg_ns_per_day",
    ])
    wall = {"last": time.perf_counter(), "cum_wall": 0.0, "cum_sim": 0.0}

    def write_csv():
        t   = dyn.get_time() / (1000 * units.fs)
        ep  = atoms.get_potential_energy()
        ek  = atoms.get_kinetic_energy()
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
            f"{t:.4f}", f"{ep:.6f}", f"{ek:.6f}", f"{tp:.1f}",
            f"{inst:.2f}", f"{avg:.2f}"
        ])

    dyn.attach(write_csv, interval=log_interval)
    dyn.run(n_steps)

    fcsv.close()
    traj.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run MD with LJ FF for multiple pack sizes"
    )
    parser.add_argument("--xyz", type=str,
                        default="BOTNet-datasets/dataset_3BPA/test_300K.xyz")
    parser.add_argument("--cell_size", type=float, default=100.0)
    parser.add_argument("--min_dist", type=float, default=2.5)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--output_dir", type=str, default="results/lj/")
    parser.add_argument("--dt_fs", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--temperature_K", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    sigma_map = {6: 3.50, 1: 2.50, 7: 3.25, 8: 3.00}
    epsilon_map = {6: 0.070, 1: 0.020, 7: 0.150, 8: 0.200}

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for N in [1, 10, 100, 1000]:
        out_dir = os.path.join(args.output_dir, f"{N}")
        print(f"=== MD: N={N} copies → {out_dir} ===")
        atoms = pack_molecules(args.xyz, N, args.cell_size, args.min_dist, rng_seed=123)
        run_md(
            atoms,
            sigma_map, epsilon_map,
            args.cutoff,
            out_dir,
            args.dt_fs,
            args.n_steps,
            args.temperature_K,
            args.friction,
            args.log_interval
        )

if __name__ == "__main__":
    main()
