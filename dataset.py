import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_df(directory, split):
    path = os.path.join(directory, split)
    files = os.listdir(path)
    list_df = []

    for file in files:
        d = dict(np.load(os.path.join(path, file)))
        d["file"] = file
        list_df.append(d)
    return pd.DataFrame.from_dict(list_df)


class TileDataset(Dataset):
    def __init__(self, data_type, source, search, data_folder, split="train"):
        self.df = load_df(os.path.join(data_folder, data_type, source), split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        config_feat = torch.tensor(row["config_feat"].astype(np.float32))
        node_feat = torch.tensor(row["node_feat"].astype(np.float32))
        node_opcode = torch.tensor(row["node_opcode"].astype(np.int64))
        edge_index = torch.tensor(np.swapaxes(row["edge_index"], 0, 1).astype(np.int64))
        target = (
            row["config_runtime"] / (row["config_runtime_normalizers"] + 1e-5)
        ).astype(
            np.float32
        )  # /row['config_runtime_normalizers']
        # minmax scale the target, we only care about order
        target = (target - np.mean(target)) / (np.std(target) + 1e-5)

        #         target = (target-np.mean(target))/(np.std(target))
        target = torch.tensor(target)
        return config_feat, node_feat, node_opcode, edge_index, target


def tile_collate_fn(batch):
    config_feat, node_feat, node_opcode, edge_index, target = zip(*batch)
    config_feat = torch.stack(config_feat)
    node_feat = torch.stack(node_feat)
    node_opcode = torch.stack(node_opcode)
    edge_index = torch.stack(edge_index)
    target = torch.stack(target)

    # only take one graph
    return {
        "config_feat": config_feat[0],
        "node_feat": node_feat[0],
        "node_opcode": node_opcode[0],
        "edge_index": edge_index[0],
        "target": target[0],
    }


class LayoutDataset(Dataset):
    def __init__(
        self, data_type, source, search, data_folder, split="train", max_configs=128
    ):
        self.df = load_df(os.path.join(data_folder, data_type, source, search), split)
        self.max_configs = max_configs
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # example shape
        # edge_index (9099, 2)
        # node_feat (5673, 140)
        # node_opcode (5673,)
        # node_config_feat (5704, 166, 18)
        # node_config_ids (166,)
        # node_splits (1, 2)
        # config_runtime (5704,)
        row = self.df.iloc[idx]
        node_feat = torch.tensor(row["node_feat"].astype(np.float32))
        node_opcode = torch.tensor(row["node_opcode"].astype(np.int64))
        edge_index = torch.tensor(np.swapaxes(row["edge_index"], 0, 1).astype(np.int64))

        # layout only
        sparse_node_config_feat = row["node_config_feat"].astype(np.float32)
        node_config_ids = row["node_config_ids"].astype(np.int64)

        target = row["config_runtime"].astype(np.float32)
        target = (target - np.mean(target)) / (np.std(target) + 1e-5)

        if sparse_node_config_feat.shape[0] < self.max_configs:
            random_indices = list(range(sparse_node_config_feat.shape[0]))
        else:
            random_indices = random.sample(
                range(sparse_node_config_feat.shape[0]), self.max_configs
            )

        sparse_node_config_feat = sparse_node_config_feat[random_indices]

        # convert node_config_feat to (num_configs, num_nodes, num_features)
        node_config_feat = (
            np.ones(
                (sparse_node_config_feat.shape[0], node_feat.shape[0], 18),
                dtype=np.float32,
            )
            * -1
        )
        node_config_feat[:, node_config_ids] = sparse_node_config_feat

        node_config_feat = torch.tensor(node_config_feat)

        target = target[random_indices]
        # minmax scale the target, we only care about order

        target = torch.tensor(target)
        return node_config_feat, node_feat, node_opcode, edge_index, target


def layout_collate_fn(batch):
    node_config_feat, node_feat, node_opcode, edge_index, target = zip(*batch)
    node_config_feat = torch.stack(node_config_feat)
    node_feat = torch.stack(node_feat)
    node_opcode = torch.stack(node_opcode)
    edge_index = torch.stack(edge_index)
    target = torch.stack(target)

    # only take one graph
    return {
        "node_config_feat": node_config_feat[0],
        "node_feat": node_feat[0],
        "node_opcode": node_opcode[0],
        "edge_index": edge_index[0],
        "target": target[0],
    }
