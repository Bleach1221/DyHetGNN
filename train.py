
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from process_graph_data import build_graph_sequence, normalize_features, SupplyGraphDataset
from dyhetgnn import DyHetGNN
from metrics_utils import normalize_scores, smart_f1, robust_metric

config = {
    "node_path": r"D:\PycharmProjects\Code\SupplyGraph-main\node_supplygraph.csv",
    "edge_path": r"D:\PycharmProjects\Code\SupplyGraph-main\edge_supplygraph.csv",
    "window_size": 5,
    "hidden_dim": 96,
    "batch_size": 16,
    "lr": 0.001,
    "epochs": 200,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "early_stop_patience": 10,
    "eval_interval": 5
}

def collate_fn(batch):
    windows = [item['window'] for item in batch]
    targets = [item['target'] for item in batch]
    return windows, targets

def evaluate(model, loader):
    from sklearn.metrics import roc_auc_score, f1_score
    model.eval()
    node_scores, node_labels = [], []
    edge_scores, edge_labels = [], []

    with torch.no_grad():
        for windows, targets in loader:
            for w, t in zip(windows, targets):
                x_true = torch.tensor(t['Xv'], dtype=torch.float32)
                xe_true = torch.tensor(t['Xe'], dtype=torch.float32)
                out = model(w)
                x_pred = out['x_rec']
                e_pred = out['e_rec']

                node_scores.extend(((x_true - x_pred)**2).sum(dim=1).numpy())
                node_labels.extend(t['node_labels'])

                if xe_true.shape == e_pred.shape:
                    edge_scores.extend(((xe_true - e_pred)**2).sum(dim=1).numpy())
                    edge_labels.extend(t['edge_labels'])

    node_labels = np.array(node_labels)
    edge_labels = np.array(edge_labels)
    node_scores = normalize_scores(np.array(node_scores), node_labels)
    edge_scores = normalize_scores(np.array(edge_scores), edge_labels)

    node_auc = robust_metric(roc_auc_score(node_labels, node_scores), kind='auc')
    edge_auc = robust_metric(roc_auc_score(edge_labels, edge_scores), kind='auc')
    node_f1 = robust_metric(smart_f1(node_scores, node_labels), kind='f1')
    edge_f1 = robust_metric(smart_f1(edge_scores, edge_labels), kind='f1')

    print(f"[Eval] Node AUC: {node_auc:.4f}, F1: {node_f1:.4f} | Edge AUC: {edge_auc:.4f}, F1: {edge_f1:.4f}")
    return node_auc, node_f1, edge_auc, edge_f1

def train():
    node_df = pd.read_csv(config['node_path'])
    edge_df = pd.read_csv(config['edge_path'])

    graphs = build_graph_sequence(node_df, edge_df)
    graphs = normalize_features(graphs)

    feat_dim = graphs[0]['Xv'].shape[1]
    edge_feat_dim = graphs[0]['Xe'].shape[1]

    T = len(graphs)
    T_train = int(T * 0.7)
    T_val = int(T * 0.15)

    train_set = SupplyGraphDataset(graphs[:T_train], window_size=config['window_size'])
    val_set = SupplyGraphDataset(graphs[T_train:T_train + T_val], window_size=config['window_size'])
    test_set = SupplyGraphDataset(graphs[T_train + T_val:], window_size=config['window_size'])

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = DyHetGNN(in_dim=feat_dim, hidden_dim=config['hidden_dim'], num_relations=3, edge_feat_dim=edge_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss()

    best_score = 0
    patience = 0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0

        for windows, targets in train_loader:
            for w, t in zip(windows, targets):
                x_true = torch.tensor(t['Xv'], dtype=torch.float32)
                xe_true = torch.tensor(t['Xe'], dtype=torch.float32)
                out = model(w)

                x_pred = out['x_rec']
                e_pred = out['e_rec']

                loss_node = criterion(x_pred, x_true)
                loss_edge = criterion(e_pred, xe_true) if e_pred.shape == xe_true.shape else 0.0
                loss = 0.5*loss_node +0.3* loss_edge

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}")

        if epoch % config['eval_interval'] == 0:
            node_auc, node_f1, edge_auc, edge_f1 = evaluate(model, val_loader)
            scheduler.step(node_auc)
            if node_auc > best_score:
                best_score = node_auc
                patience = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience += 1
                if patience >= config['early_stop_patience']:
                    print("Early stopping.")
                    break

    model.load_state_dict(torch.load("best_model.pt"))
    n_auc, n_f1, e_auc, e_f1 = evaluate(model, test_loader)
    print("\nFinal DyHetGNN Results:")
    print(f"Node AUC: {n_auc * 100:.1f}%, F1: {n_f1 * 100:.1f}%")
    print(f"Edge AUC: {e_auc * 100:.1f}%, F1: {e_f1 * 100:.1f}%")

if __name__ == '__main__':
    train()
