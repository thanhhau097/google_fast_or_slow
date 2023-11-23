import copy
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def get_obj_mbs(obj, cp=True):
    return sys.getsizeof(copy.deepcopy(obj) if cp else obj) / (1 << 20)


def prune_graph(data):
    # print("Pruning graph...")
    new_data = deepcopy(dict(data))
    # print("Original graph has {} nodes and {} edges".format(data["node_feat"].shape[0], data["edge_index"].shape[0]))
    in_edge_index = data["edge_index"][
        np.isin(data["edge_index"], data["node_config_ids"]).any(1)
    ]

    in_node_ids = np.unique(in_edge_index)
    assert len(set(data["node_config_ids"]) - set(in_node_ids)) == 0
    lookup = np.ones(data["node_feat"].shape[0]) * -1
    lookup[in_node_ids] = np.arange(in_node_ids.shape[0])

    in_node_feats = data["node_feat"][in_node_ids, :]
    in_node_opcode = data["node_opcode"][in_node_ids]
    in_edge_index = lookup[in_edge_index]
    in_node_config_ids = lookup[data["node_config_ids"]]

    new_data["node_feat"] = in_node_feats
    new_data["node_opcode"] = in_node_opcode
    new_data["edge_index"] = in_edge_index
    new_data["node_config_ids"] = in_node_config_ids
    # print("New graph has {} nodes and {} edges".format(new_data["node_feat"].shape[0], new_data["edge_index"].shape[0]))
    return new_data


def remove_dupplicated_node_configs(data):
    reshaped_config_feat = (
        data["node_config_feat"].reshape(data["node_config_feat"].shape[0], -1) + 2
    )  # avoid zeros
    positional_array = np.random.random(
        reshaped_config_feat.shape[1]
    )  # multiply each value by its position to avoid removing permutations by accident
    reshaped_values = (reshaped_config_feat * positional_array[None, :]).sum(1)
    is_equal_matrix = (
        reshaped_values[None, :] == reshaped_values[:, None]
    )  # quadratic matrix of all pairwise equalities
    # is_equal_matrix[np.triu_indices(is_equal_matrix.shape[0], 0)] = 0 # only get diagonal to avoid remove twice
    is_equal_matrix = np.tril(
        is_equal_matrix, -1
    )  # only get diagonal to avoid remove twice
    to_remove_ids = np.unique(np.where(is_equal_matrix)[0])
    # print("Removing {} duplicated node configs out of {}".format(to_remove_ids.shape[0], data["node_config_feat"].shape[0]))
    data["config_runtime"] = np.delete(data["config_runtime"], to_remove_ids)
    data["node_config_feat"] = np.delete(
        data["node_config_feat"], to_remove_ids, axis=0
    )
    return data


def find_duplicate_rows(data):
    matrix = (
        data["node_config_feat"]
        .reshape(data["node_config_feat"].shape[0], -1)
        .astype(np.int32)
    )

    # Get unique rows and inverse index
    _, unique_idx, inverse = np.unique(
        matrix, axis=0, return_index=True, return_inverse=True
    )

    # Create a dictionary of duplicates
    duplicates = {}
    for i, inv in enumerate(inverse):
        if list(np.where(inverse == inv)[0]) != [i]:
            duplicates.setdefault(unique_idx[inv], []).append(i)

    # Filter out entries with only one index (i.e., unique rows)
    dup_config_dct = {k: np.array(v) for k, v in duplicates.items() if len(v) > 1}

    all_dup_idx = [v[v != k] for k, v in dup_config_dct.items()]
    all_dup_idx = np.concatenate(all_dup_idx) if len(all_dup_idx) else []

    return dup_config_dct, all_dup_idx


def dedup_configs(data):
    dup_config_dct, all_dup_idx = find_duplicate_rows(data)

    for org_idx, idx_list in dup_config_dct.items():
        data["config_runtime"][org_idx] = round(
            np.mean(data["config_runtime"][idx_list])
        )

    if len(all_dup_idx):
        data["config_runtime"] = np.delete(data["config_runtime"], all_dup_idx)
        data["node_config_feat"] = np.delete(
            data["node_config_feat"], all_dup_idx, axis=0
        )

    return data


def test_dedup_configs(data):
    res = (
        remove_dupplicated_node_configs(copy.deepcopy(data))["node_config_feat"].shape
        == dedup_configs(copy.deepcopy(data))["node_config_feat"].shape
    )
    assert res
    return res


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


def test_compression(data, db=False):
    org = data["node_config_feat"].astype(np.int32)
    comp = compress_configs(data["node_config_feat"])
    decomp = decompress_configs(comp)

    if db:
        print(org.shape, comp.shape, decomp.shape)
        print(org[0, :2], comp[0, :2], decomp[0, :2], sep="\n")
        print(get_obj_mbs(org), get_obj_mbs(comp), get_obj_mbs(decomp))

    res = (org == decomp).all()

    assert res
    assert round(get_obj_mbs(org) / get_obj_mbs(comp)) == 6

    return res


if __name__ == "__main__":
    root = Path("./npz_all/npz")
    max_workers = 2

    for collection in ["layout/xla", "layout/nlp"]:
        for ctype in ["default", "random"]:
            dst_dir = root / f"{collection}_pruned_compressed" / ctype
            for split in ["train", "valid", "test"]:
                print("Loading {} data...".format(split))
                split_src_dir = root / collection / ctype / split
                split_dst_dir = dst_dir / split
                split_dst_dir.mkdir(parents=True, exist_ok=True)

                def _process_one_npz(npz_path):
                    out_p = split_dst_dir / npz_path.name

                    if out_p.exists():
                        return

                    data = dict(np.load(str(npz_path), allow_pickle=True))
                    data = prune_graph(data)
                    if split == "train":
                        data = remove_dupplicated_node_configs(data)
                    data["node_config_feat"] = compress_configs(
                        data["node_config_feat"]
                    )
                    np.savez_compressed(out_p, **data)

                process_map(
                    _process_one_npz,
                    list(split_src_dir.glob("*.npz")),
                    max_workers=max_workers,
                )
