import pandas as pd
import ast
import json
from rapidfuzz import fuzz

gold = pd.read_csv("RULE_GOLDEN_DATASET_TEMPLATE.csv")
pred = pd.read_csv("RULE_BEST_ATTRIBUTES.csv")

df = gold.merge(pred, on="place_id", how="inner")

# field_name : (truth_column, pred_column)
ATTRS = {
    "name":       ("truth_name", "best_name"),
    "phone":      ("truth_phone", "best_phone"),
    "address":    ("truth_address", "best_address"),
    "categories": ("truth_categories", "best_category"),   # << FIXED
    "website":    ("truth_website", "best_website"),
}

# --------------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------------

def clean_phone(p):
    if pd.isna(p): return ""
    digits = "".join([d for d in str(p) if d.isdigit()])
    return digits[-10:]

def clean_address(a):
    if pd.isna(a): return ""
    a = str(a).lower()
    return " ".join(a.split()).strip()

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

def clean_name(n):
    if pd.isna(n): return ""
    n = str(n).lower()
    n = "".join(c if c.isalnum() or c==" " else " " for c in n)
    return " ".join(n.split())

def normalize(v, field):
    if field=="phone": return clean_phone(v)
    if field=="address": return clean_address(v)
    if field=="website": return clean_website(v)
    if field=="categories": return clean_category(v)
    return clean_name(v)

# --------------------------------------------------------
# EVALUATION
# --------------------------------------------------------

print("\n=== RULE-BASED ACCURACY (NORMALIZED) ===\n")

results = []
for field, (truth_col, pred_col) in ATTRS.items():

    correct = 0
    total = 0

    for _, row in df.iterrows():

        truth = row.get(truth_col, "")
        pred  = row.get(pred_col, "")

        if pd.isna(truth) or pd.isna(pred) or truth=="" or pred=="":
            continue

        t = normalize(truth, field)
        p = normalize(pred, field)

        total += 1

        if t == p or fuzz.ratio(t, p) >= 90:
            correct += 1

    if total > 0:
        print(f"{field:10s}: {correct/total*100:.2f}%")
        results.append(correct/total*100)
    else:
        print(f"{field:10s}: N/A (no rows)")

print(f"\nOVERALL ACCURACY: {sum(results)/len(results):.2f}%\n")
