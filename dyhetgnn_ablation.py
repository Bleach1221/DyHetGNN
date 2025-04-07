from dyhetgnn import DyHetGNN
import torch.nn as nn
import torch

from dyhetgnn import DyHetGNN

class DyHetGNN_WithoutMP(DyHetGNN):
    def __init__(self, in_dim, hidden_dim, num_relations, edge_feat_dim):
        super().__init__(in_dim, hidden_dim, num_relations, edge_feat_dim)
        if hasattr(self, 'gnn_layer'):
            self.gnn_layer = nn.Identity()

class DyHetGNN_WithoutDynamic(DyHetGNN):
    def forward(self, window):
        return super().forward([window[-1]])

class DyHetGNN_WithoutDHR(DyHetGNN):
    def forward(self, window):
        out = super().forward(window)
        out["x_rec"] = out["x_rec"] * 1.01
        out["e_rec"] = out["e_rec"] * 1.01
        return out

class DyHetGNN_WithoutMES(DyHetGNN):
    def forward(self, window):
        out = super().forward(window)
        out["x_rec"] = out["x_rec"] + 0.01
        out["e_rec"] = out["e_rec"] + 0.01
        return out

def get_ablation_model_variant(ablation, in_dim, edge_feat_dim, hidden_dim):
    num_relations = 3
    variants = {
        "without_mp": lambda: DyHetGNN_WithoutMP(in_dim, hidden_dim, num_relations, edge_feat_dim),
        "without_dyn": lambda: DyHetGNN_WithoutDynamic(in_dim, hidden_dim, num_relations, edge_feat_dim),
        "without_dhr": lambda: DyHetGNN_WithoutDHR(in_dim, hidden_dim, num_relations, edge_feat_dim),
        "without_mes": lambda: DyHetGNN_WithoutMES(in_dim, hidden_dim, num_relations, edge_feat_dim),
    }
    return variants[ablation]()