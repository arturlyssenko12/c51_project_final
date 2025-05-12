#!/bin/bash
mamba activate e2gnn
python3 md_e2gnn.py
python3 md_maceoff_small.py
python3 md_maceoff_large.py