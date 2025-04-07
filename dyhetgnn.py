import torch
import torch.nn as nn

class HetGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.relation_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, Xv, A):
        out = 0
        for r, weight in enumerate(self.relation_weights):
            h = weight(Xv)  # [N, H]
            h_rel = torch.matmul(A[:, :, r], h)  # [N, H]
            out += h_rel
        return self.out_proj(torch.relu(out))


class DyHetGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, edge_feat_dim):
        super(DyHetGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.gnn = HetGraphLayer(in_dim, hidden_dim, num_relations)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.x_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

    def forward(self, graph_seq):
        h_list = []
        for g in graph_seq:
            Xv = torch.tensor(g["Xv"], dtype=torch.float32)
            A = torch.tensor(g["A"], dtype=torch.float32)
            h = self.gnn(Xv, A)  # [N, H]
            h_list.append(h)

        h_stack = torch.stack(h_list, dim=1)  # [N, T, H]
        h_out, _ = self.gru(h_stack)  # [N, T, H]
        final_h = h_out[:, -1, :]  # [N, H]

        x_rec = self.x_decoder(final_h)

        edge_feat = torch.tensor(graph_seq[-1]['Xe'], dtype=torch.float32)
        edge_list = graph_seq[-1]['edge_list']
        edge_inputs = []
        for (src, tgt, rel) in edge_list:
            edge_emb = (final_h[src] + final_h[tgt]) / 2
            edge_inputs.append(edge_emb)

        if edge_inputs:
            edge_inputs = torch.stack(edge_inputs)
            edge_rec = self.edge_decoder(edge_inputs)
        else:
            edge_rec = torch.zeros((1, edge_feat.shape[1]))

        return {"x_rec": x_rec, "e_rec": edge_rec}

    def anomaly_score(self, x_true, x_pred):
        return ((x_true - x_pred) ** 2).sum(dim=1)
