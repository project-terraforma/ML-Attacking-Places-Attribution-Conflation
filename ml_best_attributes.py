import os
import re
import json
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
GOLDEN_FILE     = "ML_GOLDEN_DATASET_TEMPLATE.csv"   # labeled golden dataset
NORMALIZED_FILE = "NORMALIZED_SOURCES.csv"           # same as rule-based input
OUTPUT_ML       = "ML_BEST_ATTRIBUTES.csv"

ATTRS      = ["name", "phone", "address", "website", "categories"]
PROVIDERS  = ["foursquare", "meta", "microsoft"]

os.makedirs("models", exist_ok=True)

# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()

def normalize_source_name(src):
    """
    Map various source names to the 3 ML providers:
      - foursquare / four_square / foursqaure -> 'foursquare'
      - msft, microsoft_structured           -> 'microsoft'
      - meta, meta_structured                -> 'meta'
    """
    if pd.isna(src):
        return ""
    s = str(src).lower().strip()
    if s in ["foursqaure", "four_square", "foursquare"]:
        return "foursquare"
    if s in ["msft", "microsoft", "microsoft_structured"]:
        return "microsoft"
    if s in ["meta", "meta_structured"]:
        return "meta"
    return s  # fallback

# =========================================================
# 1. TRAINING FEATURE BUILDER (USES GOLDEN DATASET)
# =========================================================

def build_training_features(df_golden, attr):
    """
    Build features for ONE attribute during training.

    df_golden has columns like:
      truth_<attr>_source, truth_<attr>,
      foursquare_<attr>, meta_<attr>, microsoft_<attr>

    We build ONE ROW PER PLACE, with columns:
      <provider>_sim, <provider>_exact, <provider>_present
    and label = truth_source.
    """
    rows = []

    truth_src_col = f"truth_{attr}_source"
    truth_val_col = f"truth_{attr}"

    for _, row in df_golden.iterrows():
        truth_src = row.get(truth_src_col, None)
        if pd.isna(truth_src) or truth_src == "":
            continue

        truth_val = normalize_text(row.get(truth_val_col, ""))

        feat = {"truth_source": str(truth_src)}

        for p in PROVIDERS:
            col = f"{p}_{attr}"
            raw_val = row.get(col, "")
            val = normalize_text(raw_val)

            # similarity = overlapping tokens with TRUTH (training only)
            if truth_val and val:
                sim = len(set(truth_val.split()) & set(val.split()))
            else:
                sim = 0

            exact   = int(truth_val == val)
            present = int(bool(val.strip()))

            feat[f"{p}_sim"]      = sim
            feat[f"{p}_exact"]    = exact
            feat[f"{p}_present"]  = present

        rows.append(feat)

    feat_df = pd.DataFrame(rows)
    out_file = f"ML_TRAIN_FEATURES_{attr}.csv"
    feat_df.to_csv(out_file, index=False)
    print(f"  Saved training features → {out_file}   rows={len(feat_df)}")
    return feat_df

# =========================================================
# 2. TRAIN A SINGLE ATTRIBUTE MODEL
# =========================================================

def train_one_attribute(df_golden, attr):
    print(f"\n=== TRAINING ATTRIBUTE: {attr} ===")
    data = build_training_features(df_golden, attr)

    if data.shape[0] < 5:
        print("  Not enough rows — skipping.")
        return

    y_text = data["truth_source"].astype(str)
    X = data.drop(columns=["truth_source"])

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    acc_log = accuracy_score(y_test, logreg.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    if acc_log >= acc_rf:
        best_model = logreg
        best_name  = "logreg"
        best_acc   = acc_log
    else:
        best_model = rf
        best_name  = "rf"
        best_acc   = acc_rf

    bundle = {"model": best_model, "label_encoder": le}
    out_path = f"models/{attr}_model.joblib"
    dump(bundle, out_path)

    print(f"  Logistic Regression Accuracy: {acc_log*100:.2f}%")
    print(f"  Random Forest Accuracy:      {acc_rf*100:.2f}%")
    print(f"  → Saved best model ({best_name}) with acc={best_acc*100:.2f}% → {out_path}")

def train_all_attributes():
    df_golden = pd.read_csv(GOLDEN_FILE)
    print("\nLoaded Golden Dataset with columns:")
    print(df_golden.columns.tolist())

    for attr in ATTRS:
        train_one_attribute(df_golden, attr)

    print("\nALL MODELS TRAINED.\n")

# =========================================================
# 3. BUILD INFERENCE FEATURES FROM NORMALIZED_SOURCES (LONG FORMAT)
# =========================================================

def build_groups(df_long):
    """
    df_long is NORMALIZED_SOURCES.csv, long format.
    Returns dict: place_id -> group DataFrame.
    """
    df_long = df_long.copy()
    df_long["source_norm"] = df_long["source"].apply(normalize_source_name)
    groups = {pid: g for pid, g in df_long.groupby("place_id")}
    return groups

def build_inference_features(groups, attr):
    """
    For each place_id group, build one feature vector with:
      f"{provider}_sim", f"{provider}_exact", f"{provider}_present"

    NOTE: At inference we DON'T know truth_<attr>.
    We approximate:
      - sim   = number of tokens in that provider's value
      - exact = 0 (no truth)
      - present = 1 if provider row exists and value non-empty, else 0
    """
    X_rows = []
    pid_list = []

    for pid, g in groups.items():
        feat = {}
        for p in PROVIDERS:
            sub = g[g["source_norm"] == p]

            if not sub.empty and attr in sub.columns:
                raw_val = sub.iloc[0][attr]
                val = normalize_text(raw_val)
                present = int(bool(val.strip()))
            else:
                val = ""
                present = 0

            sim   = len(val.split()) if val else 0
            exact = 0  # we don't know truth at inference

            feat[f"{p}_sim"]      = sim
            feat[f"{p}_exact"]    = exact
            feat[f"{p}_present"]  = present

        X_rows.append(feat)
        pid_list.append(pid)

    X = pd.DataFrame(X_rows)
    X.to_csv(f"ML_INFER_FEATURES_{attr}.csv", index=False)
    print(f"  Inference features for {attr} → ML_INFER_FEATURES_{attr}.csv   rows={len(X)}")
    return X, pid_list

# =========================================================
# 4. PICK ACTUAL ATTRIBUTE VALUE FROM GROUP & BEST SOURCE
# =========================================================

def pick_attr_value(group, src, attr):
    """
    group: DataFrame for one place_id (all sources)
    src:   provider name, e.g. 'meta'
    attr:  'name', 'phone', 'address', 'website', 'categories'

    If the predicted provider is missing for this place, we fall back
    to the first provider that actually exists and has a non-empty value.
    This handles cases where some providers are NULL / absent.
    """
    # 1) Try the predicted provider
    if isinstance(src, str) and src != "":
        sub = group[group["source_norm"] == src]
        if not sub.empty and attr in sub.columns:
            val = sub.iloc[0][attr]
            if pd.notna(val) and str(val).strip():
                return val

    # 2) Fallback: pick first present provider in a fixed priority order
    for p in PROVIDERS:
        sub = group[group["source_norm"] == p]
        if not sub.empty and attr in sub.columns:
            val = sub.iloc[0][attr]
            if pd.notna(val) and str(val).strip():
                return val

    # 3) Nothing available
    return ""

def majority_best_source(rec):
    """
    rec: dict with *_source fields
    """
    votes = [
        rec.get("name_source", ""),
        rec.get("phone_source", ""),
        rec.get("address_source", ""),
        rec.get("website_source", ""),
        rec.get("categories_source", ""),
    ]
    votes = [v for v in votes if isinstance(v, str) and v != ""]
    if not votes:
        return ""
    return Counter(votes).most_common(1)[0][0]

# =========================================================
# 5. RUN ML CONFLATION ON NORMALIZED_SOURCES
# =========================================================

def run_ml_conflation():
    print("\n=== RUNNING ML CONFLATION ON NORMALIZED_SOURCES ===")
    df_long = pd.read_csv(NORMALIZED_FILE)
    print(f"Loaded NORMALIZED_SOURCES with {len(df_long)} rows")

    groups = build_groups(df_long)

    # place_id -> result dict
    results = {}

    for attr in ATTRS:
        print(f"\nPredicting attribute: {attr}")
        bundle = load(f"models/{attr}_model.joblib")
        model = bundle["model"]
        le    = bundle["label_encoder"]

        X, pid_list = build_inference_features(groups, attr)
        y_pred_encoded = model.predict(X)
        y_pred_src     = le.inverse_transform(y_pred_encoded)

        for pid, src in zip(pid_list, y_pred_src):
            src = str(src)
            group = groups[pid]

            best_val = pick_attr_value(group, src, attr)

            if pid not in results:
                results[pid] = {"place_id": pid}

            if attr == "categories":
                results[pid]["categories_source"] = src
                results[pid]["best_category"]     = best_val
            else:
                results[pid][f"{attr}_source"] = src
                results[pid][f"best_{attr}"]   = best_val

    # compute overall best_source via majority vote
    for pid, rec in results.items():
        rec["best_source"] = majority_best_source(rec)

    out = pd.DataFrame(list(results.values()))

    # enforce column order similar to rule-based output
    cols_order = [
        "place_id",
        "best_source",
        "best_name",
        "best_phone",
        "best_website",
        "best_address",
        "best_category",
        "name_source",
        "phone_source",
        "website_source",
        "address_source",
        "categories_source",
    ]
    cols_order = [c for c in cols_order if c in out.columns]
    out = out[cols_order]

    out.to_csv(OUTPUT_ML, index=False)
    print(f"\nWROTE ML_BEST_ATTRIBUTES → {OUTPUT_ML} with {len(out)} places.\n")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # 1) Train models from golden dataset
    train_all_attributes()

    # 2) Run ML conflation on NORMALIZED_SOURCES.csv
    run_ml_conflation()
