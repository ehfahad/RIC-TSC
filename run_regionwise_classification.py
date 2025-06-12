# run_regionwise_classification.py

"""
Main script for region-specific classification using MiniRocket.
Supports ID (in-distribution) and OOD (out-of-distribution) evaluation.
"""

import os
import pandas as pd
from utils.region_exp_utils import (
    load_region_dataset,
    run_minirocket_pipeline,
    plot_conf_matrix,
    print_summary,
    save_results
)

# Config
DATA_DIR = "data/region_causal_datasets"
RESULTS_DIR = "results/region_specific_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 365
LABEL_COL = "label"
ALL_FEATURES = ['HV_anom', 'S2_water', 'S2_zenith', 'LS_water', 'LS_zenith', 't2m', 'r2', 'sp', 'sst']

# Prompt
print("Choose evaluation setting:")
print("1 = ID (in-distribution)")
print("2 = OOD (out-of-distribution)")
print("3 = Both")
choice = input("Enter choice [1/2/3]: ").strip()

# Regions from files
region_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_causal_timeseries.csv")]
regions = sorted([f.split("_")[0] for f in region_files])

# === ID ===
if choice in ["1", "3"]:
    results = []            # to reset results for ID evaluation

    print("\n🔍 Starting region-wise ID evaluation...")
    for region in regions:
        if region == "ALL":
            continue  # skip the ALL row
        
        print(f"\n=== REGION: {region} (ID) ===")
        df = load_region_dataset(DATA_DIR, region)

        # Causal features
        causal_cols = [c for c in df.columns if c not in ["lake_id", "date", "label", "region"]]
        acc_causal, cm_causal, le = run_minirocket_pipeline(df, SEQ_LEN, LABEL_COL, causal_cols)
        if acc_causal is None:
            print(f"⚠️ Skipping region {region} due to insufficient data.")
            continue
        print(f"✅ Causal Accuracy: {acc_causal * 100:.2f}%")
        plot_conf_matrix(cm_causal, le.classes_, f"{region} ID: Causal", f"{RESULTS_DIR}/{region}_id_causal.png")

        # === All-feature model (global features) ===
        full_df = pd.read_csv("data/processed/all_lakes_timeseries - truncated.csv", parse_dates=["date"])
        full_df = full_df.sort_values(["lake_id", "date"])
        df_all_region = full_df[full_df["region"] == region].copy()

        all_features_present = [col for col in ALL_FEATURES if col in df_all_region.columns]
        acc_all, cm_all, _ = run_minirocket_pipeline(df_all_region, SEQ_LEN, LABEL_COL, all_features_present)
        print(f"✅ All-feature Accuracy: {acc_all * 100:.2f}%")
        plot_conf_matrix(cm_all, le.classes_, f"{region} ID: All", f"{RESULTS_DIR}/{region}_id_all.png")

        results.append({
            "setting": "ID",
            "region": region,
            "acc_causal": acc_causal * 100,
            "acc_all": acc_all * 100,
            "delta": (acc_causal - acc_all) * 100,
        })

    # Save/print
    save_results(results, f"{RESULTS_DIR}/id_results.csv")
    print_summary(results, setting="ID")

# === OOD ===
if choice in ["2", "3"]:
    results = []       # to reset results for OOD evaluation

    print("\n🌍 Starting region-wise OOD evaluation...")

    # Load global causal dataset once
    df_global = load_region_dataset(DATA_DIR, "ALL")
    causal_cols = [c for c in df_global.columns if c not in ["lake_id", "date", "label", "region"]]

    for train_region in regions:
        if train_region == "ALL":
            continue  # skip the ALL row

        print(f"\n=== OOD: Train on {train_region} | Test on all others ===")

        df_train = df_global[df_global["region"] == train_region].copy()
        df_test = df_global[df_global["region"] != train_region].copy()

        # Run causal model
        acc_causal, cm_causal, le = run_minirocket_pipeline(
            train_df=df_train,
            seq_len=SEQ_LEN,
            label_col=LABEL_COL,
            feature_cols=causal_cols,
            test_df=df_test
        )

        if acc_causal is None:
            print(f"⚠️ Skipping {train_region} due to insufficient class variety.")
            continue

        print(f"✅ Causal OOD Accuracy: {acc_causal * 100:.2f}%")
        plot_conf_matrix(cm_causal, le.classes_, f"{train_region} OOD: Causal", f"{RESULTS_DIR}/{train_region}_ood_causal.png")

        # All-feature model using full dataset
        full_df = pd.read_csv("data/processed/all_lakes_timeseries - truncated.csv", parse_dates=["date"])
        full_df = full_df.sort_values(["lake_id", "date"])
        df_train_all = full_df[full_df["region"] == train_region].copy()
        df_test_all = full_df[full_df["region"] != train_region].copy()

        all_features_present = [col for col in ALL_FEATURES if col in df_train_all.columns]
        acc_all, cm_all, _ = run_minirocket_pipeline(df_train_all, SEQ_LEN, LABEL_COL, all_features_present, df_test_all)

        print(f"✅ All-feature OOD Accuracy: {acc_all * 100:.2f}%")
        plot_conf_matrix(cm_all, le.classes_, f"{train_region} OOD: All", f"{RESULTS_DIR}/{train_region}_ood_all.png")

        results.append({
            "setting": "OOD",
            "region": train_region,
            "acc_causal": acc_causal * 100,
            "acc_all": acc_all * 100,
            "delta": (acc_causal - acc_all) * 100,
        })

    # Save/print
    save_results(results, f"{RESULTS_DIR}/ood_results.csv")
    print_summary(results, setting="OOD")


print("\n🎉 Evaluation complete.")
