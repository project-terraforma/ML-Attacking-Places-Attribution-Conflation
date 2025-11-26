import pandas as pd
import json
import re
from collections import Counter
from rapidfuzz import fuzz
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # go up one folder
INPUT_NORMALIZED = BASE_DIR / "NORMALIZED_SOURCES.csv"
OUTPUT_BEST = Path(__file__).resolve().parent / "RULE_BEST_ATTRIBUTES.csv"

# Source priority used ONLY for best_source tie-breaking
# (not for excluding anyone)
SOURCE_PRIORITY = {
    "meta": 0,
    "msft": 1,
    "microsoft": 1,
    "foursquare": 2,
    "four_square": 2,
    "microsoft_structured": 3,
    "meta_structured": 3,
    "foursquare_structured": 3,
}

# Words that are junk/noisy in names (removed from canonical form)
NAME_JUNK_WORDS = {
    "hours", "address", "location", "store", "inc", "llc", "co", "company",
    "restaurant", "hotel", "casino", "atm", "center", "centre", "official",
    "site", "near", "nearby", "best", "in", "the"
}

# Phrases that indicate SEO / junk name suffixes
STOP_NAME_PHRASES = [
    "hours",
    "address",
    "official site",
    "official",
    "site",
    "near me",
    "nearby",
    "reviews",
    "location",
    "directions",
    "best ",
    " in ",
]

BAD_WEBSITE_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "twitter.com",
    "x.com",
    "bing.com",
    "yelp.com",
    "tripadvisor.com",
]

# Coarse category mapping
COARSE_CATEGORY_MAP = {
    "restaurant": "food",
    "fast_food": "food",
    "pizza": "food",
    "sandwich": "food",
    "burger": "food",
    "chicken": "food",

    "grocery": "retail",
    "supermarket": "retail",
    "retail": "retail",
    "convenience_store": "retail",
    "department_store": "retail",

    "hotel": "lodging",
    "motel": "lodging",
    "resort": "lodging",
    "lodging": "lodging",

    "car": "automotive",
    "auto": "automotive",
    "automotive": "automotive",
    "repair": "automotive",
    "dealer": "automotive",
    "gas_station": "automotive",

    "health": "medical",
    "medical": "medical",
    "clinic": "medical",
    "therapy": "medical",
    "hospital": "medical",
    "pharmacy": "medical",
    "dentist": "medical",

    "event": "entertainment",
    "venue": "entertainment",
    "casino": "entertainment",
    "theater": "entertainment",
    "cinema": "entertainment",
    "stadium": "entertainment",
    "arena": "entertainment",

    "service": "services",
    "professional": "services",
    "bank": "services",
    "financial": "services",
    "legal": "services",
    "accounting": "services",
}

# ======================================================
# SAFE JSON + CLEAN HELPERS
# ======================================================

def safe_json(val):
    """Safe JSON loader (like your normalization code)."""
    if val is None:
        return None
    val = str(val).strip()
    if val == "" or val.lower() == "nan":
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


def clean_text(x):
    """Lowercase, remove punctuation, collapse spaces."""
    if not x:
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_name_for_compare(name):
    """
    Normalize business names so tiny differences don't count:
    - lower
    - remove punctuation
    - remove trailing store IDs like #6285, 0007, -1234
    - remove junk words (hours/address/etc.)
    """
    if not name:
        return ""
    n = clean_text(name)

    # remove trailing store / numeric ids
    n = re.sub(r"(#\s*\d{1,5})\b", " ", n)
    n = re.sub(r"\bstore\s*\d{1,5}\b", " ", n)
    n = re.sub(r"\b\d{1,5}\b$", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    tokens = [t for t in n.split() if t not in NAME_JUNK_WORDS]
    return " ".join(tokens).strip()


def has_store_number(name: str) -> bool:
    """Detects explicit store numbers like '6285', '0007', etc."""
    if not name:
        return False
    return bool(re.search(r"\b\d{3,6}\b", str(name)))


def clean_phone(p):
    """
    Phone normalization:
    - strip non-digits
    - keep last 10 digits if length >= 10 (for US-style numbers)
    - otherwise keep whatever digits exist (for intl/short formats)
    """
    if not p:
        return ""
    digits = re.sub(r"\D", "", str(p))
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def extract_domain(url_value):
    """
    Normalize a single URL or the first non-empty URL in a list
    into a domain like "foo.com".
    """
    if not url_value:
        return ""
    # If it's a list, take the first non-empty
    if isinstance(url_value, list):
        first = None
        for u in url_value:
            if u and str(u).strip():
                first = str(u)
                break
        if not first:
            return ""
        u = first
    else:
        u = str(url_value)

    u = u.lower().strip()
    u = u.replace("https://", "").replace("http://", "")
    u = u.replace("www.", "")
    return u.split("/")[0].strip()


def parse_address(addr_json):
    """
    address field is list of dicts:
    [{freeform, locality, region, postcode, country}]

    Return:
      full_norm, street_norm, city_norm, state_norm, postal_norm, country_norm,
      completeness_score.
    """
    if not addr_json:
        return "", "", "", "", "", "", 0

    if isinstance(addr_json, str):
        addr_json = safe_json(addr_json)

    if not isinstance(addr_json, list) or len(addr_json) == 0:
        return "", "", "", "", "", "", 0

    a = addr_json[0] or {}

    # raw freeform to detect unit, number, etc.
    street_raw = a.get("freeform", "") or ""
    city_raw   = a.get("locality", "") or ""
    state_raw  = a.get("region", "") or ""
    postal_raw = a.get("postcode", "") or ""
    country_raw= a.get("country", "") or ""

    street  = clean_text(street_raw)
    city    = clean_text(city_raw)
    state   = clean_text(state_raw)
    postal  = clean_text(postal_raw)
    country = clean_text(country_raw)

    full = clean_text(f"{street} {city} {state} {postal} {country}")

    # completeness scoring
    has_street_num  = bool(re.search(r"\b\d+\b", str(street_raw)))
    has_street_name = bool(re.search(r"[A-Za-z]", str(street_raw)))
    has_unit        = bool(re.search(r"\b(ste|suite|unit|apt|#)\b", str(street_raw), flags=re.I))

    completeness = 0
    if has_street_num and has_street_name:
        completeness += 3
    if has_unit:
        completeness += 1
    if postal:
        completeness += 1
    if city and state:
        completeness += 1

    return full, street, city, state, postal, country, completeness


def normalize_category(cat_json):
    """
    categories field is often dict: {"primary": "...", "alternate": [...]}
    Return normalized primary string.
    """
    if not cat_json:
        return ""
    if isinstance(cat_json, str):
        cat_json = safe_json(cat_json)

    if isinstance(cat_json, dict):
        primary = cat_json.get("primary", "")
    else:
        primary = str(cat_json)

    return clean_text(primary)


def source_rank(src):
    """Lower number = better source, used only for best_source tie-break."""
    if not src:
        return 999
    s = str(src).lower().strip()
    return SOURCE_PRIORITY.get(s, 999)


def coarse_category(cat_str):
    """Map raw category string into coarse bucket."""
    if not cat_str:
        return "other"
    c = cat_str.lower()
    for key, bucket in COARSE_CATEGORY_MAP.items():
        if key in c:
            return bucket
    return "other"


# ======================================================
# RULES (one per attribute) — USING ALL SOURCES
# ======================================================

def rule_name(group):
    """
    NAME selection (uses ALL sources, structured + non-structured)

    - Normalize each candidate name (lower, strip punctuation, remove junk/store ids).
    - Down-rank noisy/SEO names (hours, address, near me, store numbers, etc.).
    - Choose name with highest average similarity across sources (token_set_ratio).
    - Tie-break: prefer shorter canonical name, then better source rank.
    """
    candidates = []  # (idx, canon_name, raw_name, base_noise, src, has_store_num)

    for idx, r in group.iterrows():
        raw = r.get("name", "")
        if not raw or str(raw).strip() == "":
            continue

        raw_str = str(raw)
        canon = normalize_name_for_compare(raw_str)
        if not canon:
            canon = clean_text(raw_str)

        src = str(r.get("source", "")).lower().strip()

        # base noise scoring
        lower_raw = raw_str.lower()
        noise = 0
        phrase_hits = sum(1 for p in STOP_NAME_PHRASES if p in lower_raw)
        noise += phrase_hits

        # "sentence-like" / SEO-like: long names, or having both hours + address
        if len(raw_str.split()) >= 8:
            noise += 1
        if "hours" in lower_raw and "address" in lower_raw:
            noise += 2

        store_flag = has_store_number(raw_str)

        candidates.append((idx, canon, raw_str, noise, src, store_flag))

    if not candidates:
        return "", ""

    # store-number penalty only if at least one source does NOT have store number
    any_without_store = any(not c[5] for c in candidates)

    # pairwise similarity on canonical names
    scores = []  # (total_score, canon_len, src_rank, idx, raw_name)
    for i, (idx_i, canon_i, raw_i, noise_i, src_i, store_i) in enumerate(candidates):
        sim_sum = 0
        count = 0
        for j, (idx_j, canon_j, raw_j, noise_j, src_j, store_j) in enumerate(candidates):
            if i == j:
                continue
            sim_sum += fuzz.token_set_ratio(canon_i, canon_j)
            count += 1
        avg_sim = sim_sum / count if count > 0 else 0

        noise = noise_i
        if any_without_store and store_i:
            # heavy penalty for store-number names when at least one clean name exists
            noise += 2  # 2 * 10 = 20 point hit

        total_score = avg_sim - noise * 10
        scores.append((total_score, len(canon_i), source_rank(src_i), idx_i, raw_i))

    # sort by: highest score, then shorter canonical name, then better source
    scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, _, _, best_idx, best_raw = scores[0]

    best_src = group.loc[best_idx, "source"]
    return best_raw, best_src


def rule_address(group):
    """
    ADDRESS selection (uses ALL sources)

    - Parse into components (street, city, state, zip, etc.).
    - Compute match_key = street + city + state for consistency.
    - Score:
        * primary: how many other sources share the same match_key
        * tie-break: higher completeness score
        * tie-break: longer full address (more specific)
    """
    candidates = []  # (match_key, full, completeness, idx)

    for idx, r in group.iterrows():
        addr_json = r.get("address", "")
        addr_json = safe_json(addr_json) if isinstance(addr_json, str) else addr_json

        full, street, city, state, postal, country, completeness = parse_address(addr_json)
        if not full:
            continue

        match_key = clean_text(f"{street} {city} {state}")
        candidates.append((match_key, full, completeness, idx))

    if not candidates:
        return "", ""

    key_counts = Counter(c[0] for c in candidates)
    max_count = max(key_counts.values())
    best_keys = {k for k, c in key_counts.items() if c == max_count}

    filtered = [c for c in candidates if c[0] in best_keys]

    # tie-break: completeness (desc), then full length (desc)
    filtered.sort(key=lambda x: (-x[2], -len(x[1])))

    best_match_key, best_full, best_comp, best_idx = filtered[0]
    best_src = group.loc[best_idx, "source"]

    return best_full, best_src


def rule_phone(group):
    """
    PHONE selection (uses ALL sources)

    - Treat each row's `phone` value (JSON string or raw) as input.
    - Run clean_phone to normalize digits.
    - Logic:
        * majority vote on the cleaned phone number (if any tie ≥2)
        * else pick phone from best-priority source.
    """
    phone_rows = []  # (cleaned_phone, idx)

    for idx, r in group.iterrows():
        raw_phone = r.get("phone", "")
        cp = clean_phone(raw_phone)
        if cp:
            phone_rows.append((cp, idx))

    if not phone_rows:
        return "", ""

    counts = Counter(pr[0] for pr in phone_rows)
    max_count = max(counts.values())
    best_nums = [n for n, c in counts.items() if c == max_count]

    # If we have a majority (count >= 2), pick among those numbers
    if max_count >= 2:
        candidates = [pr for pr in phone_rows if pr[0] in best_nums]
        candidates.sort(key=lambda x: source_rank(group.loc[x[1], "source"]))
        chosen_num, chosen_idx = candidates[0]
        chosen_src = group.loc[chosen_idx, "source"]
        return chosen_num, chosen_src

    # No majority – just pick from best-priority source
    phone_rows.sort(key=lambda x: source_rank(group.loc[x[1], "source"]))
    chosen_num, chosen_idx = phone_rows[0]
    chosen_src = group.loc[chosen_idx, "source"]
    return chosen_num, chosen_src


def rule_website(group):
    """
    WEBSITE selection (uses ALL sources, considers ALL URLs)

    - For each row, parse ALL URLs from `website` JSON list.
    - Normalize each to a domain.
    - Discard junk domains (facebook/yelp/etc.) if any better options exist.
    - Majority vote on domains (across all URLs from all sources).
    - Tie-break among URLs with that domain:
        * prefer domain containing brand tokens (from name across all sources)
        * prefer better source rank
        * prefer shorter URL (less tracking-ish).
    """
    # Collect brand tokens from NAMES across all sources
    brand_tokens = set()
    for _, r in group.iterrows():
        nm = r.get("name", "")
        canon = normalize_name_for_compare(nm)
        for t in canon.split():
            if t:
                brand_tokens.add(t)

    candidates = []  # (domain, full_url, idx)

    for idx, r in group.iterrows():
        web = r.get("website", "")
        web_json = safe_json(web) if isinstance(web, str) else web

        if isinstance(web_json, list):
            for u in web_json:
                if not u or not str(u).strip():
                    continue
                domain = extract_domain(str(u))
                if not domain:
                    continue
                candidates.append((domain, str(u), idx))
        else:
            # single string or None
            if web and str(web).strip():
                domain = extract_domain(str(web))
                if domain:
                    candidates.append((domain, str(web), idx))

    if not candidates:
        return "", ""

    # Filter out junk domains if there are any non-junk candidates
    non_junk = [c for c in candidates if not any(bad in c[0] for bad in BAD_WEBSITE_DOMAINS)]
    use_list = non_junk if non_junk else candidates

    if not use_list:
        return "", ""

    # Majority vote on domain
    domain_counts = Counter(c[0] for c in use_list)
    max_count = max(domain_counts.values())
    best_domains = [d for d, c in domain_counts.items() if c == max_count]

    # Among candidates whose domain is in best_domains, score them
    scored = []  # (score, src_rank, url_len, domain, full_url, idx)
    for domain, full_url, idx in use_list:
        if domain not in best_domains:
            continue

        src = str(group.loc[idx, "source"]).lower().strip()
        src_r = source_rank(src)

        brand_hits = sum(1 for t in brand_tokens if t and t in domain)
        # Higher brand_hits is better; shorter domain is slightly better
        domain_score = brand_hits * 5 - len(domain) * 0.05

        scored.append((domain_score, src_r, len(full_url), domain, full_url, idx))

    if not scored:
        # fallback: just pick any from majority domains
        for domain, full_url, idx in use_list:
            if domain in best_domains:
                best_src = group.loc[idx, "source"]
                return full_url or domain, best_src

    # sort by: highest score, then better source rank, then shorter URL
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    best_score, best_src_rank, best_url_len, best_domain, best_url, best_idx = scored[0]
    best_src = group.loc[best_idx, "source"]

    return best_url or best_domain, best_src


def rule_category(group):
    """
    CATEGORY selection (uses ALL sources)

    - Map categories to coarse ontology bucket.
    - Majority vote on coarse bucket.
    - Within that bucket, pick the most specific leaf:
        * more words (e.g., "chicken_restaurant" > "restaurant")
        * longer token string as tie-break.
    """
    candidates = []  # (norm_cat, coarse, idx)

    for idx, r in group.iterrows():
        cat_json = r.get("categories", "")
        norm_cat = normalize_category(cat_json)
        if not norm_cat:
            continue
        coarse = coarse_category(norm_cat)
        candidates.append((norm_cat, coarse, idx))

    if not candidates:
        return "", ""

    coarse_counts = Counter(c[1] for c in candidates)
    max_count = max(coarse_counts.values())
    best_coarse = {k for k, c in coarse_counts.items() if c == max_count}

    filtered = [c for c in candidates if c[1] in best_coarse]

    # specificity score:
    # 1) more "words" (split on '_' and spaces)
    # 2) longer string
    scored = []  # (num_words, length, norm_cat, idx)
    for norm_cat, coarse, idx in filtered:
        token_str = norm_cat.replace("_", " ")
        num_words = len(token_str.split())
        scored.append((num_words, len(norm_cat), norm_cat, idx))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    best_num_words, best_len, best_cat, best_idx = scored[0]
    best_src = group.loc[best_idx, "source"]

    return best_cat, best_src


# ======================================================
# ATTRIBUTE-LEVEL CONFLATION → BEST SOURCE
# ======================================================

def choose_best_source(attr_sources):
    """
    attr_sources is dict like:
      {"name": "meta", "phone": "msft", ...}

    Best overall source = one that "wins" most attributes.
    Tie-break by priority order (if needed).
    """
    counts = Counter([s for s in attr_sources.values() if s])
    if not counts:
        return ""

    best_count = max(counts.values())
    best_sources = [s for s, c in counts.items() if c == best_count]

    best_sources.sort(key=source_rank)
    return best_sources[0]


# ======================================================
# MAIN ENGINE
# ======================================================

def run_rule_based_conflation(input_csv=INPUT_NORMALIZED, output_csv=OUTPUT_BEST):
    df = pd.read_csv(input_csv)

    # normalize provider names a bit
    df["source"] = (
        df["source"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # normalize msft / four_square naming
    df["source"] = df["source"].replace({
        "msft": "microsoft",
        "four_square": "foursquare",
    })

    results = []

    for place_id, group in df.groupby("place_id"):
        best_name, name_src       = rule_name(group)
        best_phone, phone_src     = rule_phone(group)
        best_website, web_src     = rule_website(group)
        best_address, addr_src    = rule_address(group)
        best_category, cat_src    = rule_category(group)

        attr_sources = {
            "name": name_src,
            "phone": phone_src,
            "website": web_src,
            "address": addr_src,
            "category": cat_src,
        }

        best_source = choose_best_source(attr_sources)

        results.append({
            "place_id": place_id,
            "best_source": best_source,

            "best_name": best_name,
            "best_phone": best_phone,
            "best_website": best_website,
            "best_address": best_address,
            "best_category": best_category,

            # debugging columns
            "name_source": name_src,
            "phone_source": phone_src,
            "website_source": web_src,
            "address_source": addr_src,
            "category_source": cat_src,
        })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(out)} places.")
    return out


if __name__ == "__main__":
    run_rule_based_conflation()
