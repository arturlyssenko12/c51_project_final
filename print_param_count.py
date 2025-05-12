import torch
import os,sys
# Add E2GNN source path
sys.path.append(os.path.join(os.getcwd(), 'E2GNN'))
from E2GNN import E2GNN


# model = torch.load("/home/alyssenko/c51_project/e2gnn_student_supervised_HESSIAN.model", map_location="cuda:0")

model = torch.load("e2gnn_student_supervised_HESSIAN.model")


trainable_params = sum(p.numel() for p in model.parameters())
print(trainable_params)