#!/usr/bin/env python3

import os
import time
import pandas as pd
import torch
from ase.io import read
from tqdm import tqdm

from mace.data import config_from_atoms, AtomicData, KeySpecification
from mace.tools.utils import AtomicNumberTable

from mace.tools import torch_geometric


def evaluate_mace(
    xyz_paths,
    model_path,
    device="cuda:0",
    dt_fs=1.0,
    csv_path="mace_throughput.csv",
):
    """
    Inference loop for a MACE model, mirroring the e2gnn evaluate_full logic.
    Logs simulated ns/day based on dt_fs and writes to CSV.

    Args:
        xyz_paths:  list of .xyz file paths (frames are concatenated)
        model_path: path to your saved MACE model
        device:     e.g. "cuda:0" or "cpu"
        dt_fs:      MD timestep in femtoseconds (e.g. 1.0)
        csv_path:   where to save a one‑row CSV with column 'ns_day'
    Returns:
        e_true, e_pred, f_true, f_pred lists
    """
    # -- load the model --
    model = torch.load(model_path, map_location=device)
    model.to(device=device, dtype=torch.float32).eval()

    # -- load and concatenate all Atoms frames --
    atoms_list = []
    for p in xyz_paths:
        atoms_list.extend(read(p, index=":"))

    # -- build AtomicData list --
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    head = getattr(model, "available_heads", ["energy"])[0]

    data_list = []
    for atoms in atoms_list:
        cfg = config_from_atoms(
            atoms,
            key_specification=KeySpecification(info_keys={}, arrays_keys={}),
            head_name=head,
        )
        atomic_data = AtomicData.from_config(
            config=cfg,
            z_table=z_table,
            cutoff=cutoff,
            heads=[head],
        )
        data_list.append(atomic_data)

    # -- create PyG DataLoader --
    loader =  torch_geometric.dataloader.DataLoader(data_list, batch_size=1, shuffle=False)

    # -- inference & timing --
    e_true, e_pred = [], []
    f_true, f_pred = [], []
    n_steps = 0
    t0 = time.time()

    for batch in tqdm(loader, desc="MACE inference"):
        batch = batch.to(device=device)

        # ensure integer indices
        batch.edge_index = batch.edge_index.long()
        batch.batch      = batch.batch.long()
        if hasattr(batch, "head"):
            batch.head = batch.head.long()

        out = model(batch.to_dict())

        # energies
        e_pred.append(out["energy"].item())

        # forces
        forces = out["forces"].view(-1)
        f_pred.extend(forces.cpu().tolist())

        n_steps += 1

    wall = time.time() - t0
    sim_ns = n_steps * dt_fs * 1e-6      # fs → ns
    ns_day = sim_ns / wall * 86400.0     # ns/day

    # -- save throughput --
    pd.DataFrame([{"ns_day": ns_day}]).to_csv(csv_path, index=False)
    print(f"Processed {n_steps} frames in {wall:.1f}s → {ns_day:.2f} ns/day")

    return e_true, e_pred, f_true, f_pred


if __name__ == "__main__":
    e_t, e_p, f_t, f_p = evaluate_mace(
        xyz_paths=[
            "/home/alyssenko/c51_project/BOTNet-datasets/dataset_3BPA/test_600K.xyz"
        ],
        model_path="/home/alyssenko/c51_project/MACE-OFF23_small.model",
        device="cuda:0",
        dt_fs=1.0,
        csv_path="results_for_analysis/mace_small/mace_small_inference.csv",
    )

