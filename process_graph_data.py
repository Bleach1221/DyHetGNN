import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import os

NODE_TYPES = ["Product", "ProductionFacility", "StorageFacility"]
EDGE_TYPES = ["Product-Production", "Production-Storage", "Product-Storage"]

def build_node_index(node_df: pd.DataFrame) -> Dict[str, int]:
    unique_nodes = node_df[['node_type', 'entity_id']].drop_duplicates()
    sorted_nodes = unique_nodes.sort_values(['node_type', 'entity_id']).reset_index(drop=True)
    return {
        f"{row.node_type}_{int(row.entity_id)}": idx
        for idx, row in sorted_nodes.iterrows()
    }

def extract_nodes_by_time(node_df: pd.DataFrame, node_index: Dict[str, int]) -> Dict[str, Dict[int, Tuple[np.ndarray, int]]]:
    node_by_time = defaultdict(dict)
    for date, group in node_df.groupby("date"):
        for _, row in group.iterrows():
            key = f"{row['node_type']}_{int(row['entity_id'])}"
            if key not in node_index:
                continue
            idx = node_index[key]
            features = row.drop(['date', 'entity_id', 'node_type', 'anomaly'], errors='ignore')\
                          .fillna(0).infer_objects(copy=False).to_numpy(np.float32)
            label = int(row['anomaly'])
            node_by_time[date][idx] = (features, label)
    return node_by_time

def extract_edges_by_time(edge_df: pd.DataFrame, node_index: Dict[str, int]) -> Dict[str, List[Tuple[int, int, int, np.ndarray, int]]]:
    edge_by_time = defaultdict(list)
    for date, group in edge_df.groupby("date"):
        for _, row in group.iterrows():
            etype = row['edge_type']
            if etype not in EDGE_TYPES:
                continue
            rel_id = EDGE_TYPES.index(etype)
            eid = int(row['entity_id'])

            if etype == "Product-Production":
                src_key = f"Product_{eid % 41}"
                tgt_key = f"ProductionFacility_{eid % 25}"
            elif etype == "Production-Storage":
                src_key = f"ProductionFacility_{eid % 25}"
                tgt_key = f"StorageFacility_{eid % 13}"
            elif etype == "Product-Storage":
                src_key = f"Product_{eid % 41}"
                tgt_key = f"StorageFacility_{eid % 13}"
            else:
                continue

            if src_key not in node_index or tgt_key not in node_index:
                continue
            src = node_index[src_key]
            tgt = node_index[tgt_key]
            features = row.drop(['date', 'entity_id', 'edge_type', 'anomaly'], errors='ignore')\
                          .fillna(0).infer_objects(copy=False).to_numpy(np.float32)
            label = int(row['anomaly'])
            edge_by_time[date].append((src, tgt, rel_id, features, label))
    return edge_by_time

def build_graph_sequence(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> List[Dict]:
    node_df['date'] = pd.to_datetime(node_df['date']).dt.strftime('%Y-%m-%d')
    edge_df['date'] = pd.to_datetime(edge_df['date']).dt.strftime('%Y-%m-%d')

    node_index = build_node_index(node_df)
    node_by_time = extract_nodes_by_time(node_df, node_index)
    edge_by_time = extract_edges_by_time(edge_df, node_index)
    all_dates = sorted(set(node_by_time.keys()) & set(edge_by_time.keys()))

    num_nodes = len(node_index)
    feat_len = node_df.drop(columns=['date', 'entity_id', 'node_type', 'anomaly'], errors='ignore').shape[1]
    num_relations = len(EDGE_TYPES)
    graph_sequence = []

    for date in all_dates:
        Xv = np.zeros((num_nodes, feat_len), dtype=np.float32)
        node_labels = np.zeros(num_nodes, dtype=int)
        for nid, (feat, label) in node_by_time[date].items():
            Xv[nid, :len(feat)] = feat
            node_labels[nid] = label

        edges = edge_by_time[date]
        Xe = np.array([e[3] for e in edges], dtype=np.float32) if edges else np.zeros((1, 1))
        edge_labels = np.array([e[4] for e in edges], dtype=int) if edges else np.zeros((1,), dtype=int)

        A = np.zeros((num_nodes, num_nodes, num_relations), dtype=int)
        edge_list = []
        for src, tgt, rel, _, _ in edges:
            A[src, tgt, rel] = 1
            edge_list.append((src, tgt, rel))

        edge_index = np.array(np.nonzero(A.sum(axis=-1)))  # shape: [2, num_edges]

        graph_sequence.append({
            "date": date,
            "Xv": Xv,
            "Xe": Xe,
            "A": A,
            "node_labels": node_labels,
            "edge_labels": edge_labels,
            "edge_list": edge_list,
            "edge_index": edge_index
        })

    print(f" 构建图序列完成，共计 {len(graph_sequence)} 天图，节点总数: {num_nodes}")
    return normalize_features(graph_sequence)

def normalize_features(graph_sequence):
    scaler_node = StandardScaler()
    scaler_edge = StandardScaler()

    all_Xv = np.vstack([g['Xv'] for g in graph_sequence])
    scaler_node.fit(all_Xv)

    all_Xe = np.vstack([g['Xe'] for g in graph_sequence if g['Xe'].size > 1])
    scaler_edge.fit(all_Xe)

    for g in graph_sequence:
        g['Xv'] = scaler_node.transform(g['Xv'])
        if g['Xe'].size > 1:
            g['Xe'] = scaler_edge.transform(g['Xe'])
        else:
            g['Xe'] = np.zeros((1, scaler_edge.scale_.shape[0]))

    return graph_sequence

class SupplyGraphDataset(Dataset):
    def __init__(self, graph_sequence: List[Dict], window_size: int = 5):
        self.window_size = window_size
        self.graphs = graph_sequence

    def __len__(self):
        return max(0, len(self.graphs) - self.window_size)

    def __getitem__(self, idx):
        return {
            "window": self.graphs[idx:idx + self.window_size],
            "target": self.graphs[idx + self.window_size]
        }

if __name__ == "__main__":
    node_path = r"D:\PycharmProjects\Code\SupplyGraph-main\node_supplygraph.csv"
    edge_path = r"D:\PycharmProjects\Code\SupplyGraph-main\edge_supplygraph.csv"

    if not os.path.exists(node_path) or not os.path.exists(edge_path):
        raise FileNotFoundError(" CSV 路径无效，请检查文件是否存在")

    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    graphs = build_graph_sequence(node_df, edge_df)
    dataset = SupplyGraphDataset(graphs, window_size=5)
    print(f" PyTorch Dataset 构建成功，样本数 = {len(dataset)}")
