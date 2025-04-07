
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

def align_edge_sequence(graph_window):
    target = torch.tensor(graph_window[-1]['Xe'], dtype=torch.float32)
    E, F = target.shape
    aligned_seq = []
    for g in graph_window:
        x = torch.tensor(g['Xe'], dtype=torch.float32)
        if x.shape[0] == E:
            aligned_seq.append(x)
        else:
            padded = torch.zeros((E, F))
            copy_len = min(E, x.shape[0])
            padded[:copy_len] = x[:copy_len]
            aligned_seq.append(padded)
    return torch.stack(aligned_seq, dim=1)  # [E, T, F]

class LSTMModel(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.node_lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.edge_lstm = nn.LSTM(edge_in_dim, hidden_dim, batch_first=True)
        self.x_decoder = nn.Linear(hidden_dim, in_dim)
        self.e_decoder = nn.Linear(hidden_dim, edge_in_dim)

    def forward(self, graph_window):
        X_seq = [torch.tensor(g['Xv'], dtype=torch.float32) for g in graph_window]
        X_seq = torch.stack(X_seq, dim=1)  # [N, T, F]
        E_seq = align_edge_sequence(graph_window)  # [E, T, F]

        _, (hx, _) = self.node_lstm(X_seq)
        _, (he, _) = self.edge_lstm(E_seq)

        x_rec = self.x_decoder(hx[-1])
        e_rec = self.e_decoder(he[-1])

        return {"x_rec": x_rec, "e_rec": e_rec}

class GCNModel(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, in_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_in_dim)
        )

    def forward(self, graph_window):
        g = graph_window[-1]
        x = torch.tensor(g['Xv'], dtype=torch.float32)
        edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
        xe = torch.tensor(g['Xe'], dtype=torch.float32)

        h = F.relu(self.conv1(x, edge_index))
        x_rec = self.conv2(h, edge_index)
        e_rec = self.edge_mlp(xe)
        return {"x_rec": x_rec, "e_rec": e_rec}

class HGTModel(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.node_gat = GATConv(in_dim, hidden_dim, heads=2, concat=False)
        self.edge_gat = GATConv(edge_in_dim, hidden_dim, heads=2, concat=False)
        self.x_decoder = nn.Linear(hidden_dim, in_dim)
        self.e_decoder = nn.Linear(hidden_dim, edge_in_dim)

    def forward(self, graph_window):
        g = graph_window[-1]
        x = torch.tensor(g['Xv'], dtype=torch.float32)
        xe = torch.tensor(g['Xe'], dtype=torch.float32)
        edge_index = torch.tensor(g['edge_index'], dtype=torch.long)

        h_node = self.node_gat(x, edge_index)
        h_edge = self.edge_gat(xe, edge_index)

        x_rec = self.x_decoder(h_node)
        e_rec = self.e_decoder(h_edge)
        return {"x_rec": x_rec, "e_rec": e_rec}

class TGcnModel(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.gru_node = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.conv = GCNConv(hidden_dim, in_dim)

        self.edge_gru = nn.GRU(edge_in_dim, hidden_dim, batch_first=True)
        self.edge_dec = nn.Linear(hidden_dim, edge_in_dim)

    def forward(self, graph_window):
        X_seq = [torch.tensor(g['Xv'], dtype=torch.float32) for g in graph_window]
        X_seq = torch.stack(X_seq, dim=1)  # [N, T, F]
        E_seq = align_edge_sequence(graph_window)  # [E, T, F]
        edge_index = torch.tensor(graph_window[-1]['edge_index'], dtype=torch.long)

        h_node, _ = self.gru_node(X_seq)
        h_final = h_node[:, -1, :]  # [N, H]
        x_rec = self.conv(h_final, edge_index)

        h_edge, _ = self.edge_gru(E_seq)
        e_rec = self.edge_dec(h_edge[:, -1, :])

        return {"x_rec": x_rec, "e_rec": e_rec}

class DHGNNBaseline(nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, in_dim)
        self.gru_node = nn.GRU(in_dim, in_dim, batch_first=True)

        self.edge_gru = nn.GRU(edge_in_dim, edge_in_dim, batch_first=True)
        self.edge_fc = nn.Linear(edge_in_dim, edge_in_dim)

    def forward(self, graph_window):
        node_feats = [torch.tensor(g['Xv'], dtype=torch.float32) for g in graph_window]
        node_seq = torch.stack(node_feats, dim=1)  # [N, T, F]
        edge_seq = align_edge_sequence(graph_window)
        edge_index = torch.tensor(graph_window[-1]['edge_index'], dtype=torch.long)

        out_gru, _ = self.gru_node(node_seq)
        x_t = out_gru[:, -1, :]
        h = F.relu(self.gcn1(x_t, edge_index))
        x_rec = self.gcn2(h, edge_index)

        out_edge, _ = self.edge_gru(edge_seq)
        e_rec = self.edge_fc(out_edge[:, -1, :])
        return {"x_rec": x_rec, "e_rec": e_rec}
