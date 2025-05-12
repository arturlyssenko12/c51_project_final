#!/usr/bin/env python3

import os
import sys
import time
import random

from ase.io import read
import torch
from torch.nn import MSELoss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from ase.io.trajectory import Trajectory
import pandas as pd

# local imports
from loaders.loaders import AtomsToGraphs, collate_fn_e2gnn, move_batch, AtomsDataset
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN


def evaluate_full(loader, mdl, device, dt_fs, csv_path, hess_rows=256):
    """
    Runs through the loader, computes predictions, and logs
    throughput in ns/day based on dt_fs (timestep in fs).

    Args:
        loader:    torch DataLoader yielding batches
        mdl:       the model (already .eval())
        device:    e.g. "cuda:0"
        dt_fs:     timestep in femtoseconds (e.g. 1.0)
        csv_path:  path to write CSV with column 'ns_day'
    Returns:
        None
    """
    mdl.eval()

    n_steps = 0
    t0 = time.time()

    for batch in tqdm(loader, desc="Evaluating"):
        for data in batch.to_data_list():
            n_steps += 1
            single = Batch.from_data_list([data])
            single = move_batch(single, device, torch.float)
            single.pos = single.pos.detach().clone().requires_grad_(True)

            # forward pass
            _e_out, _f_out = mdl(single)

    wall = time.time() - t0
    sim_ns = n_steps * dt_fs * 1e-6      # fs → ns
    ns_per_day = sim_ns / wall * 86400.0

    pd.DataFrame([{"ns_day": ns_per_day}]).to_csv(csv_path, index=False)
    print(f"Processed {n_steps} steps in {wall:.1f}s → {ns_per_day:.2f} ns/day")


def combine_xyz_files(paths):
    atoms = []
    for p in paths:
        atoms.extend(read(p, ":"))
    return atoms


def combine_target_dicts(paths):
    combined = {"energy": [], "forces": [], "hessian": []}
    for p in paths:
        d = torch.load(p)
        for k in combined:
            combined[k].extend(d[k])
    # move to CPU tensors
    for k in combined:
        combined[k] = [x.cpu() for x in combined[k]]
    return combined


if __name__ == "__main__":
    device = "cuda:0"

    # load the trained model and switch to eval mode
    ema_model = torch.load(
        "e2gnn_student_supervised_HESSIAN.model",
        map_location=device
    ).eval()

    # training data (here used purely for inference throughput)
    XYZ_TRAIN ="results_for_analysis/mace_small/101/traj.traj"
    PT_TRAIN = [
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_300K.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_train_mixedT.pt",
        "BOTNet-datasets/dataset_3BPA/precomputed_training_data_test_dih.pt",
    ]

    # build dataset & loader
    train_atoms = Trajectory(XYZ_TRAIN)
    train_tgt   = combine_target_dicts(PT_TRAIN)
    all_h = torch.cat([h.flatten() for h in train_tgt['hessian']])
    hessian_std = all_h.std()

    train_ds = AtomsDataset(
        train_atoms,
        train_tgt,
        device,
        hessian_scale=hessian_std,
        plot_hist=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn_e2gnn
    )

    # run inference‐only throughput evaluation
    evaluate_full(
        loader=train_loader,
        mdl=ema_model,
        device=device,
        dt_fs=1.0,
        csv_path="results_for_analysis/e2gnn_hessian/e2gnn_hessian_inference_train_dat.csv"
    )
