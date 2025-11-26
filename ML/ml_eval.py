import pandas as pd
import ast
import json
from rapidfuzz import fuzz

GOLD = "ML_GOLDEN_DATASET_TEMPLATE.csv"
PRED = "ML_BEST_ATTRIBUTES.csv"

df_g = pd.read_csv(GOLD)
df_p = pd.read_csv(PRED)

# merge on place_id
df = df_g.merge(df_p, on="place_id", how="inner")

# --------------------------------------------------------
# ATTRIBUTE → (truth_column, predicted_column)
# --------------------------------------------------------
ATTRS = {
    "name":     ("truth_name", "best_name"),
    "phone":    ("truth_phone", "best_phone"),
    "address":  ("truth_address", "best_address"),
    "website":  ("truth_website", "best_website"),
    "category": ("truth_categories", "best_category")
}

# --------------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------------
def clean_phone(p):
    if pd.isna(p): return ""
    digits = "".join([d for d in str(p) if d.isdigit()])
    return digits[-10:]  # last 10 digits (US phones)

def clean_name(n):
    if pd.isna(n): return ""
    n = str(n).lower()
    n = "".join(c if c.isalnum() or c==" " else " " for c in n)
    return " ".join(n.split())

def clean_website(w):
    if pd.isna(w): return ""
    w = str(w).lower()
    for prefix in ["https://", "http://", "www."]:
        w = w.replace(prefix, "")
    return w.split("/")[0]

def clean_category(c):
    if pd.isna(c): return ""
    try:
        obj = json.loads(c)
        if isinstance(obj, dict):
            return obj.get("primary","").lower()
    except:
        return str(c).lower()
    return str(c).lower()

def clean_address(a):
    if pd.isna(a): return ""
    try:
        obj = json.loads(a)
        if isinstance(obj, dict):
            a = " ".join([
                obj.get("freeform",""),
                obj.get("locality",""),
                obj.get("region",""),
                obj.get("postcode",""),
                obj.get("country","")
            ])
        elif isinstance(obj, list):
            parts = []
            for block in obj:
                if isinstance(block, dict):
                    parts.append(block.get("freeform",""))
                    parts.append(block.get("locality",""))
                    parts.append(block.get("region",""))
                    parts.append(block.get("postcode",""))
                    parts.append(block.get("country",""))
            a = " ".join(parts)
    except:
        pass

    a = str(a).lower()
    a = "".join(c if c.isalnum() or c==" " else " " for c in a)
    a = " ".join(a.split())
    return a

# Dispatcher
def normalize(val, field):
    if field == "phone": return clean_phone(val)
    if field == "address": return clean_address(val)
    if field == "website": return clean_website(val)
    if field == "category": return clean_category(val)
    return clean_name(val)

# --------------------------------------------------------
# EVALUATION
# --------------------------------------------------------
print("\n=== ML VALUE-BASED ACCURACY ===\n")

acc_list = []

for field, (truth_col, pred_col) in ATTRS.items():
    correct = 0
    total = 0

    for _, row in df.iterrows():
        truth = row.get(truth_col, "")
        pred = row.get(pred_col, "")

        if pd.isna(truth) or pd.isna(pred) or truth == "" or pred == "":
            continue

        truth_n = normalize(truth, field)
        pred_n = normalize(pred, field)

        total += 1

        # exact OR fuzzy ≥90
        if truth_n == pred_n or fuzz.ratio(truth_n, pred_n) >= 90:
            correct += 1

    if total > 0:
        acc = correct / total * 100
        print(f"{field:10s}: {acc:.2f}%")
        acc_list.append(acc)
    else:
        print(f"{field:10s}: N/A (no rows available)")

print(f"\nOVERALL ACCURACY: {sum(acc_list)/len(acc_list):.2f}%\n")
