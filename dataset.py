import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import find_layout_train_files_have_same_architecture


def vec_to_int(vec: np.ndarray) -> np.ndarray:
    # Powers of 7: [1, 7, 49, 343, 2401, 16807]
    powers_of_7 = np.array([7**i for i in range(6)])
    return np.dot(vec, powers_of_7).astype(np.int32)


def int_to_vec(integers: np.ndarray) -> np.ndarray:
    # Create an empty array of shape (N, 6) to store the results
    vectors = np.empty((len(integers), 6), dtype=np.int64)

    # Divide by powers of 7 and take the remainder to find each digit
    for i in range(6):
        vectors[:, i] = integers % 7
        integers //= 7

    return vectors.astype(np.int32)


def compress_configs(node_configs):
    vecs = node_configs.reshape(-1, 6).astype(np.int32) + 1
    ints = vec_to_int(vecs)
    ints = ints.reshape(node_configs.shape[0], node_configs.shape[1], 3)
    return ints


def decompress_configs(node_configs):
    ints = node_configs.astype(np.int32).reshape(-1)
    vecs = int_to_vec(ints)
    vecs = vecs.reshape(node_configs.shape[0], -1, 18) - 1
    return vecs


def load_df(directory, split):
    path = os.path.join(directory, split)
    files = os.listdir(path)
    list_df = []

    for file in tqdm(files):
        d = dict(np.load(os.path.join(path, file)))
        d["file"] = file
        if len(d["config_runtime"]) <= 1:
            print(f"skipping {file} as it only contains {len(d['config_runtime'])} configs")
            continue
        list_df.append(d)
    return pd.DataFrame.from_dict(list_df)


class TileDataset(Dataset):
    def __init__(self, data_type, source, search, data_folder, split="train", **kwargs):
        self.df = load_df(os.path.join(data_folder, data_type, source), split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        config_feat = torch.tensor(row["config_feat"].astype(np.float32))
        node_feat = torch.tensor(row["node_feat"].astype(np.float32))
        node_opcode = torch.tensor(row["node_opcode"].astype(np.int64))
        edge_index = torch.tensor(np.swapaxes(row["edge_index"], 0, 1).astype(np.int64))
        target = (row["config_runtime"] / (row["config_runtime_normalizers"] + 1e-5)).astype(
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
        self,
        data_type,
        source,
        search,
        data_folder,
        split="train",
        max_configs=64,
        scaler=None,
        tgt_scaler=None,
        use_compressed=True,
        data_concatenation=False,
        select_close_runtimes=False,
        select_close_runtimes_prob=0.5,
        filter_random_configs=False,
        valid_file_names="",
        test_file_names="",
        **kwargs,
    ):
        if search == "mix":
            # in mix mode, we load all the data both from default and random
            if not use_compressed:
                default_df = load_df(
                    os.path.join(data_folder, data_type, source, "default"), split
                )
                random_df = load_df(os.path.join(data_folder, data_type, source, "random"), split)
            else:
                default_df = load_df(
                    os.path.join(data_folder, data_type, source + "_compressed", "default"), split
                )
                random_df = load_df(
                    os.path.join(data_folder, data_type, source + "_compressed", "random"), split
                )

            # only keep random configs that has runtime inside range of default configs
            if split == "train" and filter_random_configs:
                print("Filtering random configs")
                filtered_random_df = []
                for file in tqdm(default_df["file"].unique()):
                    default_runtime = default_df[default_df["file"] == file][
                        "config_runtime"
                    ].values[0]
                    file_df = random_df.loc[random_df["file"] == file]

                    # filter out node_config_feat and runtime that are not in range of min and max default runtime
                    min_runtime = min(default_runtime)
                    max_runtime = max(default_runtime)

                    new_dict = {}
                    for col in file_df.columns:
                        if col == "config_runtime":
                            filtered_runtimes = []
                            filtered_node_config_feat = []
                            for runtime, node_config_feat in zip(
                                file_df[col].values[0], file_df["node_config_feat"].values[0]
                            ):
                                if min_runtime <= runtime <= max_runtime:
                                    filtered_runtimes.append(runtime)
                                    filtered_node_config_feat.append(node_config_feat)

                            # --------- EXPERIMENTAL CODE ---------
                            # we need to upsampling to distribution of default runtime
                            # split the default runtime to k bins, then for each bin, we sample from the filtered runtime
                            k = 128
                            bins = np.linspace(min_runtime, max_runtime, k + 1)
                            bin_indices = np.digitize(filtered_runtimes, bins)
                            bin_indices = np.array(bin_indices)
                            bin_indices = bin_indices - 1
                            bin_indices = bin_indices.tolist()

                            # for each bin, we sample runtimes from the filtered runtimes
                            sampled_runtimes = []
                            sampled_node_config_feat = []

                            for i in range(k):
                                bin_indices_i = np.where(np.array(bin_indices) == i)[0]
                                if len(bin_indices_i) > 0:
                                    # num_samples is number of default runtimes in bin i
                                    num_samples = np.sum(
                                        np.logical_and(
                                            default_runtime < bins[i + 1],
                                            default_runtime >= bins[i],
                                        )
                                    )
                                    sampled_indices = np.random.choice(
                                        bin_indices_i,
                                        # num_samples
                                        min(num_samples, len(bin_indices_i)),
                                        # len(bin_indices_i) if num_samples > 0 else 0,
                                    )
                                    sampled_runtimes.extend(sampled_indices)
                                    sampled_node_config_feat.extend(sampled_indices)

                            sampled_runtimes = np.array(filtered_runtimes)[sampled_runtimes]
                            sampled_node_config_feat = np.array(filtered_node_config_feat)[
                                sampled_node_config_feat
                            ]

                            # new_dict[col] = [np.array(filtered_runtimes)]
                            # new_dict["node_config_feat"] = [np.array(filtered_node_config_feat)]
                            new_dict[col] = [np.array(sampled_runtimes)]
                            new_dict["node_config_feat"] = [np.array(sampled_node_config_feat)]
                        else:
                            new_dict[col] = file_df[col].values

                    filtered_random_df.append(pd.DataFrame.from_dict(new_dict))

                random_df = pd.concat(filtered_random_df)

            default_df["search"] = "default"
            random_df["search"] = "random"

            self.df = pd.concat([default_df, random_df])

            # group by file, mix configs
            if split == "train" and not data_concatenation:
                self.df = (
                    self.df.groupby("file")
                    .agg(
                        {
                            "node_feat": "first",
                            "node_opcode": "first",
                            "edge_index": "first",
                            "node_config_feat": lambda x: np.concatenate(x.tolist(), axis=0),
                            "node_config_ids": "first",
                            "config_runtime": lambda x: np.concatenate(x.tolist(), axis=0),
                            "search": "first",
                        }
                    )
                    .reset_index()
                )
        else:
            if not use_compressed:
                self.df = load_df(os.path.join(data_folder, data_type, source, search), split)
            else:
                self.df = load_df(
                    os.path.join(data_folder, data_type, source + "_compressed", search), split
                )

            self.df["search"] = search

        # filter valid/test files to finetune the model
        if not valid_file_names and not test_file_names:
            pass
        else:
            val_mapping, test_mapping, test_to_val_mapping = find_layout_train_files_have_same_architecture(
                data_folder, source, search
            )
            if valid_file_names:
                if split == "train":
                    all_valid_file_names = valid_file_names.split(",")
                    filtered_train_files = []
                    for file_name in all_valid_file_names:
                        filtered_train_files.extend(val_mapping[file_name])
                    
                    if len(filtered_train_files) > 0:
                        self.df = self.df[self.df["file"].isin(filtered_train_files)]
                    else:
                        raise ValueError("No train files found for this valid file")
                elif split == "valid":
                    self.df = self.df[self.df["file"].isin(valid_file_names.split(","))]
            elif test_file_names:
                if split == "train":
                    all_test_file_names = test_file_names.split(",")
                    filtered_train_files = []
                    for file_name in all_test_file_names:
                        filtered_train_files.extend(test_mapping[file_name])
                    
                    if len(filtered_train_files) > 0:
                        self.df = self.df[self.df["file"].isin(filtered_train_files)]
                    else:
                        raise ValueError("No train files found for this test file")

                # find val files that have the same edge_index as test files
                if split == "valid":
                    all_test_file_names = test_file_names.split(",")
                    filtered_valid_files = []
                    for file_name in all_test_file_names:
                        filtered_valid_files.extend(test_to_val_mapping[file_name])
                    
                    if len(filtered_valid_files) > 0:
                        self.df = self.df[self.df["file"].isin(filtered_valid_files)]
                    else:
                        raise ValueError("No valid files found for this test file")

        self.scaler = scaler
        self.tgt_scaler = tgt_scaler
        if self.scaler is not None:
            self.scaler = self.scaler.fit(np.concatenate(self.df["node_feat"].tolist())[:, :134])
        if self.tgt_scaler is not None:
            self.tgt_scaler = self.tgt_scaler.fit(
                np.concatenate(self.df["config_runtime"].tolist())[:, None]
            )
        self.max_configs = max_configs
        self.split = split
        self.select_close_runtimes = select_close_runtimes
        self.select_close_runtimes_prob = select_close_runtimes_prob

        # break dataset into batch size chunks
        if self.split in ["valid", "valid_dedup", "test"]:
            new_df = []
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                nb_splits = int(np.ceil(row["node_config_feat"].shape[0] / max_configs))
                all_node_cfg_feat_chunks = np.array_split(row["node_config_feat"], nb_splits)
                all_runtime_chunks = np.array_split(row["config_runtime"], nb_splits)
                for subset_node_cfg_feat, subset_runtime in zip(
                    all_node_cfg_feat_chunks, all_runtime_chunks
                ):
                    new_df.append(
                        {
                            "file": row["file"],
                            "node_config_feat": subset_node_cfg_feat,
                            "node_feat": row["node_feat"],
                            "node_opcode": row["node_opcode"],
                            "edge_index": row["edge_index"],
                            "node_config_ids": row["node_config_ids"],
                            "config_runtime": subset_runtime,
                            "search": row["search"],
                        }
                    )
            self.df = pd.DataFrame.from_dict(new_df)

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
        node_feat = row["node_feat"].astype(np.float32)
        node_layout_feat = node_feat[:, 134:]
        node_feat = node_feat[:, :134]

        node_opcode = torch.tensor(row["node_opcode"].astype(np.int64))
        edge_index = torch.tensor(np.swapaxes(row["edge_index"], 0, 1).astype(np.int64))

        # layout only
        # sparse_node_config_feat = row["node_config_feat"].astype(np.int8)
        sparse_node_config_feat = row["node_config_feat"]  # .astype(np.int8)
        node_config_ids = row["node_config_ids"].astype(np.int64)

        target = row["config_runtime"].astype(np.float32)
        # target = (target - np.mean(target)) / (np.std(target) + 1e-5)

        if self.split in ["valid", "valid_dedup", "test"]:
            random_indices = list(range(sparse_node_config_feat.shape[0]))
        elif sparse_node_config_feat.shape[0] <= self.max_configs:
            random_indices = list(range(sparse_node_config_feat.shape[0]))
        else:
            if self.select_close_runtimes:
                if np.random.rand() > self.select_close_runtimes_prob:
                    random_indices = random.sample(
                        range(sparse_node_config_feat.shape[0]), self.max_configs
                    )
                else:
                    sorted_indices = np.argsort(target)

                    # select a list of k * max_configs indices then randomly select max_configs indices
                    k = np.random.randint(1, 5)
                    if k * self.max_configs < len(sorted_indices):
                        start_idx = np.random.randint(
                            0, len(sorted_indices) - k * self.max_configs
                        )
                    else:
                        start_idx = 0

                    end_idx = start_idx + k * self.max_configs
                    random_indices = sorted_indices[start_idx:end_idx]
                    random_indices = random.sample(random_indices.tolist(), self.max_configs)
            else:
                random_indices = random.sample(
                    range(sparse_node_config_feat.shape[0]), self.max_configs
                )

        sparse_node_config_feat = sparse_node_config_feat[random_indices]
        sparse_node_config_feat = decompress_configs(sparse_node_config_feat).astype(np.int8)

        node_config_feat = torch.tensor(sparse_node_config_feat, dtype=torch.long)

        target = target[random_indices]
        # minmax scale the target, we only care about order

        # normalisation
        node_feat = self.scaler.transform(node_feat)
        target = self.tgt_scaler.transform(target[:, None]).squeeze(1)

        node_feat = torch.tensor(node_feat)
        node_layout_feat = torch.tensor(node_layout_feat, dtype=torch.long)
        target = torch.tensor(target)
        return (
            node_config_feat,
            node_feat,
            node_layout_feat,
            node_opcode,
            edge_index,
            torch.tensor(node_config_ids),
            target,
        )


def layout_collate_fn(batch):
    (
        node_config_feat,
        node_feat,
        node_layout_feat,
        node_opcode,
        edge_index,
        node_config_ids,
        target,
    ) = zip(*batch)
    node_config_feat = torch.stack(node_config_feat)[0]
    node_feat = torch.stack(node_feat)[0]
    node_layout_feat = torch.stack(node_layout_feat)[0]
    node_opcode = torch.stack(node_opcode)[0]
    edge_index = torch.stack(edge_index)[0]
    node_config_ids = torch.stack(node_config_ids)[0]
    target = torch.stack(target)[0]

    # only take one graph
    return {
        "node_config_feat": node_config_feat,
        "node_feat": node_feat,
        "node_layout_feat": node_layout_feat,
        "node_opcode": node_opcode,
        "edge_index": edge_index,
        "node_config_ids": node_config_ids,
        "target": target,
    }
