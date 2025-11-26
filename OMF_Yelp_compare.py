import json
import pandas as pd
from rapidfuzz import fuzz
import re

# ======================================================
# CLEANING HELPERS
# ======================================================

def clean_text(x):
    if not x:
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def extract_domain(urls):
    if not urls or len(urls) == 0:
        return ""
    try:
        u = urls[0].lower()
        u = u.replace("https://", "").replace("http://", "")
        u = u.replace("www.", "")
        return u.split("/")[0]
    except:
        return ""


def clean_phone(p):
    if not p:
        return ""
    p = re.sub(r"\D", "", p)
    return p[-10:] if len(p) >= 10 else p


def parse_address(addr_list):
    if not addr_list or len(addr_list) == 0:
        return "", "", "", "", "", ""

    a = addr_list[0]
    street  = clean_text(a.get("freeform", "") or "")
    city    = clean_text(a.get("locality", "") or "")
    state   = clean_text(a.get("region", "") or "")
    postal  = clean_text(a.get("postcode", "") or "")
    country = clean_text(a.get("country", "") or "")

    full = clean_text(f"{street} {city} {state} {postal}")

    return full, street, city, state, postal, country


# ======================================================
# LOAD NORMALIZED OMF (NOW WITH CATEGORY + WEBSITE + SOCIALS)
# ======================================================

def load_omf(path):
    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():

        # website
        try:
            website_list = json.loads(r["website"]) if pd.notna(r["website"]) else []
        except:
            website_list = []

        # socials
        try:
            socials_list = json.loads(r["socials"]) if pd.notna(r["socials"]) else []
        except:
            socials_list = []

        # categories
        try:
            cats = json.loads(r["categories"]) if pd.notna(r["categories"]) else []
        except:
            cats = []

        # address
        try:
            addr_list = json.loads(r["address"]) if pd.notna(r["address"]) else []
        except:
            addr_list = []

        full_addr, street, city, state, postal, country = parse_address(addr_list)

        rows.append({
            "place_id": r["place_id"],
            "source": r["source"],

            "name": clean_text(r["name"]),
            "phone": clean_phone(str(r["phone"])),
            "domain": extract_domain(website_list),
            "addr": full_addr,
            "street": street,
            "city": city,
            "state": state,
            "postal": postal,
            "country": country,

            # NEW IMPORTANT FIELDS
            "categories": json.dumps(cats),
            "website": json.dumps(website_list),
            "socials": json.dumps(socials_list),
        })

    return pd.DataFrame(rows)


# ======================================================
# LOAD YELP — NOW WITH CATEGORIES
# ======================================================

def load_yelp(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue

            street  = clean_text(obj.get("address", ""))
            city    = clean_text(obj.get("city", ""))
            state   = clean_text(obj.get("state", ""))
            postal  = clean_text(obj.get("postal_code", ""))
            full = clean_text(f"{street} {city} {state} {postal}")

            rows.append({
                "business_id": obj.get("business_id", ""),
                "name": clean_text(obj.get("name", "")),
                "phone": clean_phone(obj.get("phone", "")),
                "categories": clean_text(str(obj.get("categories", ""))),
                "street": street,
                "city": city,
                "state": state,
                "postal": postal,
                "addr": full
            })

    return pd.DataFrame(rows)


# ======================================================
# SIMILARITY FUNCTIONS — ONLY NAME + ADDRESS
# ======================================================

def name_score(a, b):
    if not a or not b:
        return 0
    return fuzz.token_sort_ratio(a, b)

def addr_score(a, b):
    if not a or not b:
        return 0
    return fuzz.token_sort_ratio(a, b)

def weighted_score(omf_row, yelp_row):
    ns = name_score(omf_row["name"], yelp_row["name"])
    ad = addr_score(omf_row["addr"], yelp_row["addr"])
    return 0.65 * ns + 0.35 * ad


# ======================================================
# VALIDATION
# ======================================================

def validate(omf_df, yelp_df, show_examples=True):
    matchable = 0
    valid = 0
    valid_rows = []

    shown = 0

    for _, omf in omf_df.iterrows():
        block = yelp_df[yelp_df["city"] == omf["city"]]
        if len(block) == 0:
            continue

        best_score = -1
        best_row = None

        for _, y in block.iterrows():

            if omf["phone"] and omf["phone"] == y["phone"]:
                best_row = y
                best_score = 100
                break

            s = weighted_score(omf, y)
            if s > best_score:
                best_score = s
                best_row = y

        if best_score >= 55:
            matchable += 1

        if best_score >= 75 and best_row is not None:
            valid += 1

            valid_rows.append({
                "omf_place_id": omf["place_id"],
                "omf_source": omf["source"],

                "omf_name": omf["name"],
                "omf_address": omf["addr"],
                "omf_phone": omf["phone"],
                "omf_categories": omf["categories"],
                "omf_website": omf["website"],
                "omf_socials": omf["socials"],

                "yelp_business_id": best_row["business_id"],
                "yelp_name": best_row["name"],
                "yelp_address": best_row["addr"],
                "yelp_phone": best_row["phone"],
                "yelp_categories": best_row["categories"],

                "match_score": best_score
            })

    return len(omf_df), matchable, valid, valid_rows


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    omf = load_omf("NORMALIZED_SOURCES.csv")
    yelp = load_yelp("yelp_academic_dataset_business.json")

    total, matchable, valid, valid_rows = validate(omf, yelp, show_examples=True)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total OMF: {total}")
    print(f"Matchable: {matchable}")
    print(f"Valid: {valid}")
    acc = (valid / matchable) * 100 if matchable else 0

    pd.DataFrame(valid_rows).to_csv("VALID_MATCHES.csv", index=False)
    print("\nWrote VALID_MATCHES.csv")
