import numpy as np
from ast import literal_eval
import json
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def score_tile_max(predictions, runtime):
    score = 0
    for i in range(len(runtime)):
        preds_idx = np.argsort(predictions[i])[:50]
        predbest = np.min(runtime[i][preds_idx[:5]])
        best = np.min(runtime[i])
        score += 2 - predbest / best
    score /= len(runtime)
    return score


gt_rts = []
pred_rts = []


in_path = Path("./outputs_csv").resolve()
out_path = Path("./outputs_csv_merge").resolve()
out_path.mkdir(exist_ok=True)

for f in tqdm(list(Path("./outputs_csv").glob("*.csv"))):
    preds = pd.read_csv(f)

    # gts = [np.array(literal_eval(rt)) for rt in preds.runtime.tolist()]
    pts = [np.array(literal_eval(rt)) for rt in preds.logits.tolist()]

    # normalize predictions before mean
    # pts = [rankdata(pt) / len(pt) for pt in pts]

    # gt_rts.append(gts)
    pred_rts.append(pts)

pred_rts_mean = [
    np.mean(np.stack([rts_list[jj] for rts_list in pred_rts]), 0)
    for jj in range(len(pred_rts[0]))
]

# print(score_tile_max(pred_rts_mean, gt_rts[0]))


prediction_indices = [
    ";".join([str(int(e)) for e in np.argsort(rts)[:5]]) for rts in pred_rts_mean
]

submission_df = pd.DataFrame.from_dict(
    {
        "ID": preds.ID,
        "TopConfigs": prediction_indices,
    }
)

submission_df.to_csv(out_path / "sub_tile.csv", index=False)
