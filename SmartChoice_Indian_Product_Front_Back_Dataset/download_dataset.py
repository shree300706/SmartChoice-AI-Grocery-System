#!/usr/bin/env python3
"""
Match the 852 products in metadata/packaged_foods_india.csv to Open Food Facts
and download front + back-label images where available.

"Back" is selected as an ingredients image first, then a nutrition-label image,
because Open Food Facts does not consistently label a separate literal back photo.
The manifest records exactly which image types were found. It never fabricates images.
Resume-safe: existing downloaded files are kept and the manifest is updated after each row.
"""
import argparse, os, re, time, json, sys
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "metadata" / "packaged_foods_india.csv"
MANIFEST = ROOT / "metadata" / "image_manifest.csv"
IMGROOT = ROOT / "images"
API = "https://world.openfoodfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "SmartChoiceIndianProductDataset/1.0 (educational project; contact: add-your-email)"}
TIMEOUT = 30

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

def image_url(product, kind):
    # Prefer explicit legacy URLs, which are easy to handle consistently.
    aliases = {
        "front": ["image_front_url", "image_url"],
        "ingredients": ["image_ingredients_url"],
        "nutrition": ["image_nutrition_url"],
        "back": ["image_back_url"],
    }
    for key in aliases[kind]:
        val = product.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    # selected_images commonly looks like {front:{display:{en:url}, small:{en:url}}}
    selected = product.get("selected_images") or {}
    obj = selected.get(kind) or {}
    if isinstance(obj, dict):
        for size in ("display", "400", "large", "full", "small", "thumb"):
            block = obj.get(size)
            if isinstance(block, dict):
                for lang in ("en", "hi", "hi-in", "fr", "world"):
                    val = block.get(lang)
                    if isinstance(val, str) and val.startswith("http"):
                        return val
                for val in block.values():
                    if isinstance(val, str) and val.startswith("http"):
                        return val
    # Older API shape: image keys such as front_en / ingredients_en / nutrition_en
    images = product.get("images") or {}
    prefixes = {"front": ("front_en", "front_hi", "front"), 
                "ingredients": ("ingredients_en", "ingredients_hi", "ingredients"),
                "nutrition": ("nutrition_en", "nutrition_hi", "nutrition"),
                "back": ("back_en", "back_hi", "back")}
    for key in prefixes[kind]:
        item = images.get(key)
        if isinstance(item, dict) and item.get("rev"):
            code = str(product.get("code") or "")
            if len(code) < 13:
                code = code.zfill(13)
            if len(code) >= 13:
                folder = f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}"
                return f"https://images.openfoodfacts.org/images/products/{folder}/{key}.{item['rev']}.400.jpg"
        if isinstance(item, str) and item.startswith("http"):
            return item
    return None

def score_product(row, product):
    name = norm(row.get("Item name", ""))
    brand = norm(row.get("Brand_Name", ""))
    pname = norm(product.get("product_name") or product.get("product_name_en") or product.get("generic_name") or "")
    pbrand = norm(product.get("brands") or "")
    name_score = SequenceMatcher(None, name, pname).ratio() if name and pname else 0
    # Exact brand is a strong signal, but do not require it (OFF may have different spelling).
    brand_score = 1.0 if brand and (brand in pbrand or pbrand in brand) else (SequenceMatcher(None, brand, pbrand).ratio() if brand and pbrand else 0)
    return 0.78 * name_score + 0.22 * brand_score

def search_products(session, row):
    terms = f'{row.get("Brand_Name","")} {row.get("Item name","")}'.strip()
    params = {
        "search_terms": terms, "search_simple": 1, "action": "process",
        "json": 1, "page_size": 8,
        "fields": "code,product_name,product_name_en,generic_name,brands,countries_tags,selected_images,images,image_front_url,image_ingredients_url,image_nutrition_url,image_url,url"
    }
    r = session.get(API, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    products = data.get("products") or []
    # Prefer products tagged as India, but retain other results if the tag is absent.
    india = [p for p in products if any("india" in str(c).lower() for c in (p.get("countries_tags") or []))]
    candidates = india or products
    candidates.sort(key=lambda p: score_product(row, p), reverse=True)
    if not candidates:
        return None, 0.0
    return candidates[0], score_product(row, candidates[0])

def download(session, url, path):
    if not url:
        return False
    if path.exists() and path.stat().st_size > 1000:
        return True
    try:
        r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200 or len(r.content) < 1000:
            return False
        # Keep actual image responses only.
        ctype = r.headers.get("Content-Type", "").lower()
        if "image" not in ctype and not r.content[:3] == b"\xff\xd8\xff" and not r.content.startswith(b"\x89PNG"):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        return True
    except requests.RequestException:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=852, help="Maximum source rows to process (default: all 852)")
    ap.add_argument("--delay", type=float, default=2.5, help="Delay between product searches in seconds")
    ap.add_argument("--retry-failed", action="store_true", help="Retry rows previously marked no_match or partial")
    ap.add_argument("--min-match-score", type=float, default=0.52, help="Minimum fuzzy-match score, 0-1")
    args = ap.parse_args()
    if not CSV.exists():
        print(f"Missing dataset: {CSV}")
        sys.exit(1)
    source = pd.read_csv(CSV).fillna("")
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST).fillna("")
    else:
        man = pd.DataFrame()
    if len(man) != len(source):
        man = pd.DataFrame([{
            "dataset_row": r.get("S.No", i+1), "product_name": r.get("Item name",""),
            "brand": r.get("Brand_Name",""), "category": r.get("Category",""),
            "sub_category": r.get("Sub_Category",""), "barcode": "", "open_food_facts_product_name": "",
            "match_status": "not_searched", "front_image_path": "", "back_image_path": "",
            "ingredients_image_path": "", "nutrition_image_path": "", "front_image_url": "",
            "back_image_url": "", "ingredients_image_url": "", "nutrition_image_url": "",
            "source_url": "", "notes": ""
        } for i, r in source.iterrows()])
    session = requests.Session()
    processed = 0
    for i, row in source.iterrows():
        if processed >= args.limit:
            break
        m = man.loc[i]
        if m.get("match_status") == "complete" and not args.retry_failed:
            continue
        if m.get("match_status") in ("no_match", "partial") and not args.retry_failed:
            continue
        print(f"[{i+1}/{len(source)}] {row.get('Brand_Name','')} — {row.get('Item name','')}", flush=True)
        try:
            product, score = search_products(session, row)
        except Exception as e:
            man.at[i, "match_status"] = "search_error"
            man.at[i, "notes"] = str(e)[:250]
            man.to_csv(MANIFEST, index=False)
            time.sleep(max(args.delay, 3))
            processed += 1
            continue
        if not product or score < args.min_match_score:
            man.at[i, "match_status"] = "no_match"
            man.at[i, "notes"] = f"No reliable Open Food Facts match (best score: {score:.2f})."
            man.to_csv(MANIFEST, index=False)
            time.sleep(args.delay)
            processed += 1
            continue
        code = str(product.get("code") or "")
        pslug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(row.get("S.No", i+1)) + "_" + str(row.get("Brand_Name","")) + "_" + str(row.get("Item name","")))[:100]
        product_dir = IMGROOT / pslug
        urls = {k: image_url(product, k) for k in ("front", "ingredients", "nutrition", "back")}
        paths = {}
        for kind in ("front", "ingredients", "nutrition", "back"):
            if urls[kind]:
                out = product_dir / f"{kind}.jpg"
                if download(session, urls[kind], out):
                    paths[kind] = str(out.relative_to(ROOT)).replace("\\", "/")
        # "Back" means a real back-label view when supplied, otherwise label close-up.
        back_kind = "back" if paths.get("back") else ("ingredients" if paths.get("ingredients") else ("nutrition" if paths.get("nutrition") else None))
        man.at[i, "barcode"] = code
        man.at[i, "open_food_facts_product_name"] = product.get("product_name") or product.get("product_name_en") or ""
        man.at[i, "source_url"] = product.get("url") or (f"https://world.openfoodfacts.org/product/{code}" if code else "")
        for kind in ("front", "ingredients", "nutrition"):
            man.at[i, f"{kind}_image_url"] = urls.get(kind) or ""
            man.at[i, f"{kind}_image_path"] = paths.get(kind) or ""
        man.at[i, "back_image_url"] = urls.get(back_kind) or ""
        man.at[i, "back_image_path"] = paths.get(back_kind) or ""
        count = sum(1 for k in ("front", "ingredients", "nutrition", "back") if paths.get(k))
        man.at[i, "match_status"] = "complete" if paths.get("front") and back_kind else ("partial" if count else "matched_no_images")
        man.at[i, "notes"] = f"Fuzzy match score {score:.2f}; back view source: {back_kind or 'not available'}. Verify product identity before using."
        man.to_csv(MANIFEST, index=False)
        time.sleep(args.delay)
        processed += 1
    print(f"Finished this run. Manifest: {MANIFEST}")
    print("Check match_status and visually verify matches; not every product will have front/back images.")

if __name__ == "__main__":
    main()
