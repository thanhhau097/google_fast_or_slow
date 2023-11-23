import pandas as pd
from pathlib import Path
import glob


def merge_csvs(input_path, output_path):
    all_files = glob.glob(input_path + "/*.csv")
    merged_df = pd.concat([pd.read_csv(f) for f in all_files])
    merged_df.to_csv(output_path, index=False)

    merged_df["ID"] = merged_df["ID"].str.replace("_pruned", "")
    merged_df["ID"] = merged_df["ID"].str.replace("_compressed", "")

    assert len(merged_df) == 894
    assert len(merged_df[merged_df["ID"].str.startswith("tile:xla:")]) == 844
    assert len(merged_df[merged_df["ID"].str.startswith("layout:xla:default:")]) == 8
    assert len(merged_df[merged_df["ID"].str.startswith("layout:xla:random:")]) == 8
    assert len(merged_df[merged_df["ID"].str.startswith("layout:nlp:default:")]) == 17
    assert len(merged_df[merged_df["ID"].str.startswith("layout:nlp:random:")]) == 17


# Paths
in_path = "./outputs_csv_merge"
out_path = "./submission.csv"

# Merge CSVs
merge_csvs(in_path, out_path)
