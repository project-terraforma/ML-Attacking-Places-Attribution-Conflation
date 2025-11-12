import csv
import json

INPUT = "project_b_samples_2k.csv"
OUTPUT = "NORMALIZED_SOURCES.csv"


def safe_json(val):
    if not val or val.strip() == "":
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


def stringify(x):
    if x is None:
        return ""
    return f"=\"{str(x)}\""


with open(INPUT, encoding="utf-8") as fin, open(OUTPUT, "w", newline="", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    writer.writerow([
        "place_id",
        "source",
        "record_id",
        "update_time",
        "name",
        "categories",
        "phone",
        "website",
        "socials",
        "address",
        "confidence"
    ])

    for row in reader:

        place_id = row["id"]

        # -------------------------------------------------------
        # META
        # -------------------------------------------------------
        meta_sources = safe_json(row["sources"])
        meta_name = safe_json(row["names"])
        meta_cat = safe_json(row["categories"])
        meta_web = safe_json(row["websites"])
        meta_socials = safe_json(row["socials"])
        meta_phone = safe_json(row["phones"])
        meta_addr = safe_json(row["addresses"])
        meta_conf = row["confidence"]

        # find META record_id + update_time from "sources"
        meta_entry = None
        if isinstance(meta_sources, list):
            for item in meta_sources:
                if item.get("dataset") == "meta":
                    meta_entry = item
                    break

        writer.writerow([
            place_id,
            "meta",
            stringify(meta_entry.get("record_id") if meta_entry else None),
            meta_entry.get("update_time") if meta_entry else "",
            meta_name.get("primary") if isinstance(meta_name, dict) else "",
            json.dumps(meta_cat) if meta_cat else "",
            json.dumps(meta_phone) if meta_phone else "",
            json.dumps(meta_web) if meta_web else "",
            json.dumps(meta_socials) if meta_socials else "",
            json.dumps(meta_addr) if meta_addr else "",
            meta_conf
        ])

        # -------------------------------------------------------
        # MSFT_EXISTENCE — from "sources" where dataset="msft"
        # -------------------------------------------------------
        if isinstance(meta_sources, list):
            for item in meta_sources:
                if item.get("dataset") == "msft":
                    writer.writerow([
                        place_id,
                        "msft_existence",
                        stringify(item.get("record_id")),
                        item.get("update_time"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    ])


        # ------------------------------------------------------
        # MSFT_STRUCTURED — from base_sources + base_* fields
        # ------------------------------------------------------
        msft_struct_sources = safe_json(row["base_sources"])
        msft_name = safe_json(row["base_names"])
        msft_cat = safe_json(row["base_categories"])
        msft_web = safe_json(row["base_websites"])
        msft_socials = safe_json(row["base_socials"])
        msft_phone = safe_json(row["base_phones"])
        msft_addr = safe_json(row["base_addresses"])
        msft_conf = row["base_confidence"]

        msft_struct = None
        if isinstance(msft_struct_sources, list) and len(msft_struct_sources) > 0:
            msft_struct = msft_struct_sources[0]  # always one structured entry

        writer.writerow([
            place_id,
            "msft_structured",
            stringify(msft_struct.get("record_id") if msft_struct else None),
            msft_struct.get("update_time") if msft_struct else "",
            msft_name.get("primary") if isinstance(msft_name, dict) else "",
            json.dumps(msft_cat) if msft_cat else "",
            json.dumps(msft_phone) if msft_phone else "",
            json.dumps(msft_web) if msft_web else "",
            json.dumps(msft_socials) if msft_socials else "",
            json.dumps(msft_addr) if msft_addr else "",
            msft_conf
        ])
