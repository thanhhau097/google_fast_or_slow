from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


root = Path("./npz_all_pad/npz")
NUM_FOLDS = 9

for collection in ["layout/xla", "layout/nlp"]:
    for ctype in ["default", "random"]:
        dst_dir = root / f"{collection}_compressed_kfold" / ctype
        for split in ["train", "valid", "test"]:
            print("Loading {} data...".format(split))
            split_src_dir = root / (collection + "_compressed") / ctype / split
            split_dst_dir = dst_dir / split
            split_dst_dir.mkdir(parents=True, exist_ok=True)

            for npz_path in tqdm(list(split_src_dir.glob("*.npz"))):
                out_p = split_dst_dir / npz_path.name

                if out_p.exists():
                    continue

                data = dict(np.load(str(npz_path), allow_pickle=True))

                if split == "test":
                    np.savez_compressed(out_p, **data)
                    continue

                new_data = {}
                for k, v in data.items():
                    if k not in ["config_runtime", "node_config_feat"]:
                        new_data[k] = v

                config_runtime = data["config_runtime"]
                node_config_feat = data["node_config_feat"]

                # random index
                # set seed
                np.random.seed(42)
                idx = np.random.permutation(config_runtime.shape[0])
                for i in range(NUM_FOLDS):
                    start = i * config_runtime.shape[0] // NUM_FOLDS
                    end = (i + 1) * config_runtime.shape[0] // NUM_FOLDS if i != NUM_FOLDS - 1 else None
                    new_data["config_runtime"] = config_runtime[idx[start:end]]
                    new_data["node_config_feat"] = node_config_feat[idx[start:end]]

                    fold_name = npz_path.name.replace(".npz", f"@{i}.npz")
                    np.savez_compressed(split_dst_dir / fold_name, **new_data)
