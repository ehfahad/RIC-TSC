# generate_causal_datasets.py

import os
import pandas as pd
from utils.causal_dataset_utils import (
    load_causal_parents,
    extract_regionwise_data,
    save_region_datasets
)

CAUSAL_PARENT_CSV = "./causality/Causal_Parents_of_HV_anom_by_Region.csv"
INPUT_CSV = "./data/processed/all_lakes_timeseries - truncated.csv"
OUTPUT_DIR = "./data/region_causal_datasets"

def main():
    print("Loading full dataset...")
    df = pd.read_csv(INPUT_CSV)

    # Ensure proper datetime and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["lake_id", "date"]).copy()

    # Pre-compute dummy variables
    df["t-dummy"] = df["date"].dt.dayofyear
    df["s-dummy"] = df["lake_id"].astype("category").cat.codes
    df["r-dummy"] = df["region"].astype("category").cat.codes

    print("Loading causal parents...")
    region_parents = load_causal_parents(CAUSAL_PARENT_CSV)

    print("Extracting regionwise causal datasets...")
    for region, parent_map in region_parents.items():
        print(f"Processing region: {region}")
        causal_df = extract_regionwise_data(df, region, parent_map)
        save_region_datasets(causal_df, region, OUTPUT_DIR)

    print("✅ All region datasets saved under:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
