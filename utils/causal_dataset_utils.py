# causal_dataset_utils.py

import os
import pandas as pd
from collections import defaultdict

def load_causal_parents(csv_path):
    """
    Loads simplified causal parents file (format: A:-1|-2;B:0).
    Returns: dict {region: {feature: [lags]}}
    """
    df = pd.read_csv(csv_path)
    region_parents = {}

    for _, row in df.iterrows():
        region = row["Region"]
        raw_parents = row["CausalParents"]
        parents = defaultdict(list)

        for entry in raw_parents.split(";"):
            if not entry.strip():
                continue
            try:
                var, lags = entry.strip().split(":")
                var = var.replace(".", "_")  # Match feature names
                lags = [int(lag.strip()) for lag in lags.split("|")]
                parents[var.strip()] = lags
            except ValueError:
                print(f"[WARN] Skipping malformed entry in {region}: '{entry}'")

        region_parents[region] = dict(parents)

    return region_parents

def _build_lagged_feature_map(df, feature, lags):
    """
    Return DataFrame with specified lags for a feature.
    Positive lag value means looking back in time (t - lag).
    """
    out = pd.DataFrame(index=df.index)
    for lag in lags:
        if lag == 0:
            continue
        col_name = f"{feature}_lag{abs(lag)}"
        out[col_name] = df.groupby("lake_id")[feature].shift(abs(lag))
    return out

def extract_regionwise_data(df, region, parent_map):
    """
    Given full DataFrame, extract causal input features for one region.
    """
    df_region = df.copy() if region == "ALL" else df[df["region"] == region].copy()
    df_region = df_region.sort_values(["lake_id", "date"])

    id_vars = ["lake_id", "date", "label", "region"]
    feature_dfs = [df_region[id_vars].copy()]

    for var, lags in parent_map.items():
        if var in df_region.columns:
            if any(lag != 0 for lag in lags):
                feature_dfs.append(_build_lagged_feature_map(df_region, var, lags))
            if 0 in lags:
                feature_dfs.append(df_region[[var]])
        else:
            # Handle dummy variables explicitly
            if var == "t-dummy":
                df_region["t-dummy"] = pd.to_datetime(df_region["date"]).dt.dayofyear
                feature_dfs.append(df_region[["t-dummy"]])
            elif var == "s-dummy":
                df_region["s-dummy"] = df_region["lake_id"].astype("category").cat.codes
                feature_dfs.append(df_region[["s-dummy"]])
            elif var == "r-dummy":
                df_region["r-dummy"] = df_region["region"].astype("category").cat.codes
                feature_dfs.append(df_region[["r-dummy"]])
            else:
                print(f"[WARN] {var} not found in dataset for region {region} — skipping!")

    df_out = pd.concat(feature_dfs, axis=1)
    df_out = df_out.sort_values(["lake_id", "date"]).reset_index(drop=True)
    return df_out

def save_region_datasets(df, region, out_dir):
    """
    Save final per-region dataset to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{region}_causal_timeseries.csv"), index=False)
