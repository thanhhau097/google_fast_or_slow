import json
import os

import numpy
from tqdm import tqdm


def find_layout_train_files_have_same_architecture(
    data_folder, source, search, kfold=False
):
    save_folder = os.path.join(data_folder, "architecture_data")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # find architecture of test set file based on edge_index
    if kfold:
        suffix = "compressed_kfold"
    else:
        suffix = "compressed"
    train_folder = f"{data_folder}/layout/{source}_{suffix}/{search}/train"
    val_folder = f"{data_folder}/layout/{source}_{suffix}/{search}/valid"
    test_folder = f"{data_folder}/layout/{source}_{suffix}/{search}/test"

    train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder)]
    val_files = [os.path.join(val_folder, f) for f in os.listdir(val_folder)]
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]

    # create a dict of filename: edge_index
    train_files_dict = {}
    for f in tqdm(train_files):
        train_files_dict[os.path.basename(f)] = numpy.load(f)["edge_index"]

    val_files_dict = {}
    for f in tqdm(val_files):
        val_files_dict[os.path.basename(f)] = numpy.load(f)["edge_index"]

    # # find val files that have the same edge_index as train files
    # val_mapping = {}
    # for f in tqdm(val_files):
    #     edge_index = numpy.load(f)["edge_index"]
    #     for k, v in train_files_dict.items():
    #         if edge_index.shape[0] == v.shape[0] and (edge_index == v).all():
    #             if os.path.basename(f) not in val_mapping:
    #                 val_mapping[os.path.basename(f)] = []
    #             val_mapping[os.path.basename(f)].append(k)

    # find test files that have the same edge_index as train files
    test_mapping = {}
    for f in tqdm(test_files):
        edge_index = numpy.load(f)["edge_index"]
        for k, v in {**train_files_dict, **val_files_dict}.items():
            if edge_index.shape[0] == v.shape[0] and (edge_index == v).all():
                if os.path.basename(f) not in test_mapping:
                    test_mapping[os.path.basename(f)] = []
                test_mapping[os.path.basename(f)].append(k)

    # test_to_val_mapping = {}
    # for f in tqdm(test_files):
    #     edge_index = numpy.load(f)["edge_index"]
    #     for k, v in val_files_dict.items():
    #         if edge_index.shape[0] == v.shape[0] and (edge_index == v).all():
    #             if os.path.basename(f) not in test_to_val_mapping:
    #                 test_to_val_mapping[os.path.basename(f)] = []
    #             test_to_val_mapping[os.path.basename(f)].append(k)

    # for file, list_similar in test_files_dict.items():
    #     print(os.path.basename(file), len(list_similar))

    # print("Source:", source, "Search:", search)
    # print("count/total", len(test_files_dict), len(test_files))

    # for k, v in test_files_dict.items():
    #     print(k, v)

    # print("-----------" * 10)
    # val_json_path = os.path.join(save_folder, f"{source}_{search}_valid.json")
    # with open(val_json_path, "w") as f:
    #     json.dump(val_mapping, f, indent=4)

    # test_json_path = os.path.join(save_folder, f"{source}_{search}_test.json")
    # with open(test_json_path, "w") as f:
    #     json.dump(test_mapping, f, indent=4)

    # test_to_val_json_path = os.path.join(save_folder, f"{source}_{search}_test_to_valid.json")
    # with open(test_to_val_json_path, "w") as f:
    #     json.dump(test_to_val_mapping, f, indent=4)

    # return val_mapping, test_mapping, test_to_val_mapping
    return test_mapping


if __name__ == "__main__":
    data_folder = "/home/thanh/google_fast_or_slow/data/npz_all_pad/npz"

    for source in ["xla", "nlp"]:
        for search in ["random", "default"]:
            print("Source:", source, "Search:", search)
            (
                val_files_dict, test_files_dict, test_to_val_files_dict
            ) = find_layout_train_files_have_same_architecture(
                data_folder, source, search
            )
            print("-" * 100)
