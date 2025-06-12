# utils/region_exp_utils.py

"""
Utility functions for region-specific MiniRocket classification pipeline.
Handles preprocessing, sequence building, model training, and evaluation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sktime.transformations.panel.rocket import MiniRocket

SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def load_region_dataset(data_dir, region):
    """
    Load the causal time series CSV for a given region.
    Sort by lake and date. Fill missing values (due to lag) forward and backward.
    """
    path = os.path.join(data_dir, f"{region}_causal_timeseries.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["lake_id", "date"])
    df = df.groupby("lake_id").apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    return df


def build_dataset(df, feature_cols, label_col, seq_len=365, label_encoder=None):
    """
    Build 3D array (n_samples, n_channels, n_timepoints) for MiniRocket from lake-wise groups.
    """
    X, y = [], []
    lake_ids = df["lake_id"].unique()

    for lake_id in lake_ids:
        group = df[df["lake_id"] == lake_id]
        if len(group) != seq_len:
            continue
        X.append(group[feature_cols].reset_index(drop=True))
        y.append(group[label_col].iloc[0])

    if not X:
        return None, None, None, 0, 0

    X_np = np.stack([x.to_numpy() for x in X])
    X_np = np.transpose(X_np, (0, 2, 1))  # shape: (n_instances, n_channels, n_timepoints)

    valid_mask = ~np.isnan(X_np).any(axis=(1, 2))
    X_np = X_np[valid_mask]
    y = np.array(y)[valid_mask]
    dropped = np.sum(~valid_mask)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)

    return X_np, y_encoded, label_encoder, dropped, len(lake_ids)


def run_minirocket_pipeline(train_df, seq_len, label_col, feature_cols, test_df=None):
    """
    MiniRocket + RidgeClassifier pipeline using specific feature columns.
    Handles missing columns in test_df by filling them with zeros.
    """
    # Build training set
    X_train, y_train, le, _, _ = build_dataset(train_df, feature_cols, label_col, seq_len)
    if X_train is None or len(np.unique(y_train)) < 2:
        return None, None, None

    if test_df is not None:
        # No need to fill missing features — global dataset has consistent columns

        X_test, y_test, _, _, _ = build_dataset(test_df, feature_cols, label_col, seq_len, le)
        if X_test is None or len(np.unique(y_test)) < 2:
            return None, None, None
    else:
        # ID setting: train/test split from train_df
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
        )

    # MiniRocket pipeline
    rocket = MiniRocket(random_state=SEED)
    rocket.fit(X_train)
    X_train_trans = rocket.transform(X_train)
    X_test_trans = rocket.transform(X_test)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_trans, y_train)
    y_pred = clf.predict(X_test_trans)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm, le


def plot_conf_matrix(cm, classes, title, filename):
    """
    Save confusion matrix with consistent formatting.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📊 Saved confusion matrix: {filename}")


def save_results(results, path):
    """
    Save evaluation results to CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"📄 Results saved to {path}")


def print_summary(results, setting="ID"):
    """
    Print region-wise performance summary based on setting.
    """
    print(f"\n=== {setting.upper()} PERFORMANCE SUMMARY ===")
    if setting.upper() == "ID":
        print(f"{'Region':<10}{'Causal Acc (%)':<18}{'All Acc (%)':<18}{'Δ Causal - All (%)':<20}")
        print("-" * 66)
        for r in results:
            print(f"{r['region']:<10}{r['acc_causal']:<18.2f}{r['acc_all']:<18.2f}{r['delta']:<20.2f}")
    elif setting.upper() == "OOD":
        print(f"{'Train Region':<15}{'Causal Acc (%)':<18}{'All Acc (%)':<18}{'Δ Causal - All (%)':<20}")
        print("-" * 72)
        for r in results:
            print(f"{r['region']:<15}{r['acc_causal']:<18.2f}{r['acc_all']:<18.2f}{r['delta']:<20.2f}")
