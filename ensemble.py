from collections import defaultdict
import os
import json

import numpy as np
import pandas as pd


# TODO: update paths to folders contains predictions.npy here
fold_folders = []

fold_predictions_dict = {}

for folder in fold_folders:
    base_prediction = np.load(os.path.join(folder, "predictions.npy"), allow_pickle=True).item()
    base_prediction_dict = {file: prob for file, prob in zip(base_prediction["prediction_files"], base_prediction["predictions_probs"])}

    final_dict = {}
    for file, prob in base_prediction_dict.items():
        final_dict[file] = prob
    
    fold_predictions_dict[os.path.basename(folder)] = final_dict

mean_predictions_dict = defaultdict(list)

for file in fold_predictions_dict[list(fold_predictions_dict.keys())[0]].keys():
    for key in fold_predictions_dict.keys():
        mean_predictions_dict[file].append(fold_predictions_dict[key][file])

mean_predictions_dict = {file: np.mean(np.array(probs), axis=0) for file, probs in mean_predictions_dict.items()}
# get argsort
mean_predictions_dict = {file: np.argsort(probs) for file, probs in mean_predictions_dict.items()}

df = pd.DataFrame.from_dict({
    "ID": mean_predictions_dict.keys(),
    "TopConfigs": mean_predictions_dict.values()
})
df["TopConfigs"] = df["TopConfigs"].apply(lambda x: ";".join([str(i) for i in x]))
df.to_csv("ensemble.csv", index=False)
