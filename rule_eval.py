import pandas as pd
import ast
from rapidfuzz import fuzz

gold = pd.read_csv("RULE_GOLDEN_DATASET_TEMPLATE.csv")
pred = pd.read_csv("RULE_BEST_ATTRIBUTES.csv")

# merge the two datasets based on place_id
df = gold.merge(pred, on="place_id", how="inner")

ATTRS = ["name", "phone", "address", "category", "website"]

def clean_phone(p):
    if pd.isna(p): return ""
    digits = "".join([d for d in str(p) if d.isdigit()])
    return digits[-10:]  # last 10 digits for US phones

def clean_address(addr):
    if pd.isna(addr): return ""
    addr = str(addr).lower()
    # remove zip+4 or trailing numbers
    return " ".join(addr.split()[:-1]).strip()

def clean_website(url):
    if pd.isna(url): return ""
    url = str(url).lower()

    for prefix in ["https://", "http://", "www."]:
        url = url.replace(prefix, "")

    return url.split("/")[0]  # keep domain only

def clean_category(cat):
    if pd.isna(cat): return ""
    if isinstance(cat, str) and cat.startswith("{"):
        try:
            d = ast.literal_eval(cat)
            return d.get("primary", "").lower()
        except:
            return cat.lower()
    return str(cat).lower()

def normalize(a, field):
    if field == "phone": return clean_phone(a)
    if field == "address": return clean_address(a)
    if field == "website": return clean_website(a)
    if field == "category": return clean_category(a)
    return str(a).lower().strip()

def evaluate(row, field):
    pred = normalize(row[f"best_{field}"], field)
    truth = normalize(row[f"truth_{field}"], field)

    # exact match or fuzzy ≥ 90
    if pred == truth:
        return 1

    if fuzz.ratio(pred, truth) >= 90:
        return 1

    return 0

print("\n=== RULE-BASED ACCURACY (NORMALIZED) ===\n")

for attr in ATTRS:
    df[f"correct_{attr}"] = df.apply(lambda r: evaluate(r, attr), axis=1)
    acc = df[f"correct_{attr}"].mean() * 100
    print(f"{attr:10s}: {acc:.2f}%")

overall = df[[f"correct_{a}" for a in ATTRS]].values.mean() * 100
print(f"\nOVERALL ACCURACY: {overall:.2f}%")
