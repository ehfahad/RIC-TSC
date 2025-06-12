# run_global_classification.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sktime.transformations.panel.rocket import MiniRocket
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

SEQ_LEN = 365
LABEL_COL = "label"
ALL_FEATURES = ['HV_anom', 'S2_water', 'S2_zenith', 'LS_water', 'LS_zenith', 't2m', 'r2', 'sp', 'sst']
OUT_DIR = "results/global_classification"
os.makedirs(OUT_DIR, exist_ok=True)

def build_dataset(df, feature_cols, label_col="label", seq_len=365, le=None):
    X, y = [], []
    lake_ids = df["lake_id"].unique()

    for lake_id in lake_ids:
        group = df[df["lake_id"] == lake_id]
        if len(group) != seq_len:
            continue
        X.append(group[feature_cols].reset_index(drop=True))
        y.append(group[label_col].iloc[0])

    if not X:
        return None, None, None

    X_np = np.stack([x.to_numpy() for x in X])
    X_np = np.transpose(X_np, (0, 2, 1))

    y = np.array(y)
    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    valid_mask = ~np.isnan(X_np).any(axis=(1, 2))
    return X_np[valid_mask], y_encoded[valid_mask], le

def run_minirocket(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    rocket = MiniRocket(random_state=SEED)
    rocket.fit(X_train)
    X_train_trans = rocket.transform(X_train)
    X_test_trans = rocket.transform(X_test)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_trans, y_train)
    y_pred = clf.predict(X_test_trans)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    return acc, cm, y_test, y_pred, precision, recall, f1


def plot_conf_matrix(cm, labels, title, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"📊 Saved confusion matrix to {path}")

### === Run on global causal dataset ===
df_causal = pd.read_csv("data/region_causal_datasets/ALL_causal_timeseries.csv", parse_dates=["date"])
df_causal = df_causal.groupby("lake_id", group_keys=False).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
causal_cols = [c for c in df_causal.columns if c not in ["lake_id", "date", "label", "region"]]
X_causal, y_causal, le = build_dataset(df_causal, causal_cols, LABEL_COL, SEQ_LEN)

print(f"\n=== Full Dataset Evaluation: Causal Features ===")
acc_causal, cm_causal, _, _, prec_causal, rec_causal, f1_causal = run_minirocket(X_causal, y_causal)
print(f"✅ Accuracy (Causal): {acc_causal * 100:.2f}%")
print(f"Precision: {prec_causal:.2f} | Recall: {rec_causal:.2f} | F1: {f1_causal:.2f}")
plot_conf_matrix(cm_causal, le.classes_, "Global Causal Features", f"{OUT_DIR}/global_causal_cm.png")


### === Run on all features ===
df_all = pd.read_csv("data/processed/all_lakes_timeseries - truncated.csv", parse_dates=["date"])
df_all = df_all.sort_values(["lake_id", "date"])
X_all, y_all, _ = build_dataset(df_all, ALL_FEATURES, LABEL_COL, SEQ_LEN, le)

print(f"\n=== Full Dataset Evaluation: All Features ===")
acc_all, cm_all, _, _, prec_all, rec_all, f1_all = run_minirocket(X_all, y_all)
print(f"✅ Accuracy (All Features): {acc_all * 100:.2f}%")
print(f"Precision: {prec_all:.2f} | Recall: {rec_all:.2f} | F1: {f1_all:.2f}")
plot_conf_matrix(cm_all, le.classes_, "Global All Features", f"{OUT_DIR}/global_all_cm.png")

### === Summary ===
print("\n=== GLOBAL CLASSIFICATION SUMMARY ===")
print(f"{'Feature Set':<20}{'Acc (%)':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")
print("-" * 70)
print(f"{'Causal Features':<20}{acc_causal * 100:<12.2f}{prec_causal:<12.2f}{rec_causal:<12.2f}{f1_causal:<12.2f}")
print(f"{'All Features':<20}{acc_all * 100:<12.2f}{prec_all:<12.2f}{rec_all:<12.2f}{f1_all:<12.2f}")

### Save results
results = {
    "feature_set": ["Causal Features", "All Features"],
    "accuracy": [acc_causal * 100, acc_all * 100],
    "precision": [prec_causal, prec_all],
    "recall": [rec_causal, rec_all],
    "f1_score": [f1_causal, f1_all]
}
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/global_classification_results.csv", index=False)
print(f"📄 Results saved to {OUT_DIR}/global_classification_results.csv")