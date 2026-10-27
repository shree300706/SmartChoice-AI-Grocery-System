# SmartChoice Indian Product Front/Back Image Dataset

This kit is pre-aligned to the **852 rows** in your `packaged_foods_india.csv` source dataset.

## What is included
- `metadata/packaged_foods_india.csv` — your original product/nutrition dataset.
- `metadata/image_manifest.csv` — one row per source product, with front/back/ingredients/nutrition image path and URL fields.
- `download_dataset.py` — searches Open Food Facts for each product, fuzzy-matches name + brand, then downloads available product views.
- `images/` — starts empty; images are downloaded when you run the script.

## Important about front and back images
The original CSV does not include barcodes or product image URLs, so image matching must be done against an external catalogue. This script does **not** invent or synthesize packaging photos. Availability varies. A literal back photo is not consistently tagged in Open Food Facts; where absent, the kit uses a real ingredients-label image as the `back_image_path`, then nutrition-label image as a fallback. The manifest records the source type and match score so you can inspect it.

## Run on Windows
Open PowerShell in this extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --default-timeout=120 --retries=10 -r requirements.txt
.\.venv\Scripts\python.exe download_dataset.py --limit 50 --delay 3
```

Start with 50 to check quality. Then process the remaining rows:

```powershell
.\.venv\Scripts\python.exe download_dataset.py --limit 852 --delay 3
```

If you have already run it and want to retry rows that had no match or only partial images:

```powershell
.\.venv\Scripts\python.exe download_dataset.py --limit 852 --delay 3 --retry-failed
```

The script saves `image_manifest.csv` after every product and can resume. If Open Food Facts returns HTTP 503, wait and run again with a longer delay (for example `--delay 7`). Large-scale images are better obtained through the Open Food Facts image dataset; this row-wise matcher is a practical starter workflow for the exact product list.

## Dataset organization
Images are stored as:
`images/<source-row>_<brand>_<product>/front.jpg`
`images/<source-row>_<brand>_<product>/ingredients.jpg`
`images/<source-row>_<brand>_<product>/nutrition.jpg`
`images/<source-row>_<brand>_<product>/back.jpg` (only when a separate back view exists)

## Quality and licensing
Open Food Facts product images are community-contributed and coverage/accuracy are not guaranteed. Review the image against the product name/brand before using it for OCR or model training. Follow Open Food Facts image licensing (CC BY-SA) and attribution requirements. Product database contents have separate licensing terms; review the official documentation before redistribution.
