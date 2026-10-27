# SmartChoice: AI-assisted OCR scan and compare

## What it does
- Scan two products independently from nutrition/ingredient label images.
- OCR extracts candidate nutrition values; users can edit them before comparison.
- Personalized ranking uses dietary goals, ingredients to avoid, diet preference, optional sugar/sodium limits and budget.
- Explainable AI (XAI): per-product score contributions and global Random Forest feature importance.
- Generative AI: optional OpenAI LLM writes a grounded comparison using product facts and local TF-IDF retrieved knowledge; without an API key, local retrieval fallback is used.
- Displays a recommendation score, ML score, warnings and a side-by-side comparison table.
- Uses the included 852-product CSV to train a scikit-learn RandomForest demonstration model.
- Member 3 module: local TF-IDF retrieval from `data/knowledge_base.csv`, with optional OpenAI LLM explanation when `OPENAI_API_KEY` is configured.

## Windows setup
Open PowerShell in the folder containing `app.py`. Use:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```
Install Tesseract OCR separately on Windows and add `tesseract.exe` to PATH. If OCR cannot find Tesseract, configure `pytesseract.pytesseract.tesseract_cmd` in `app.py`.
Open http://127.0.0.1:5000.

## Team members
- Member 1: dataset, OCR and testing.
- Member 2: RandomForest demo model, ranking, evaluation and XAI contribution/feature-importance views.
- Member 3: `rag.py`, local knowledge base, TF-IDF retrieval and optional OpenAI Generative AI explanation.

## Limitations
OCR may misread decimals, digits, units or columns; verify every extracted value. Both products should be compared on the same nutrition basis (e.g. per 100 g). The demo model is trained on synthetic rule-generated labels because the CSV has no human-rated suitability labels; its metrics are not clinical accuracy. Ingredient keyword checks cannot guarantee allergy safety. This is informational, not medical advice.


## Explainable AI (XAI)
Each product result includes a transparent rule-score breakdown. Positive contributions raise the rule score; negative contributions lower it. The page also displays global Random Forest feature importance. Feature importance describes overall model behavior and is not a causal explanation of an individual prediction. The contribution breakdown explains the heuristic component; the final score also uses the Random Forest estimate.

## Generative AI
If `OPENAI_API_KEY` is configured, the app sends the comparison facts and retrieved knowledge snippets to the selected OpenAI model to generate a plain-language explanation. Set the key in PowerShell before launching the app, for example `$env:OPENAI_API_KEY="your-key"`. Keep the key private and do not hard-code it. If the key is missing or the API call fails, local RAG guidance is shown instead. Generated explanations can still make mistakes; verify all nutrition and ingredient values.


## AI recommendation chatbot
The `/api/chat` endpoint and chatbot panel let users ask for catalogue-backed suggestions in natural language. Lightweight intent detection recognizes goals such as low sugar, higher protein/fiber, lower sodium, vegetarian/vegan preferences and common “avoid …” requests. Candidates are ranked with the project's rule-based preference score and demonstration Random Forest, with local TF-IDF knowledge retrieval (RAG). If `OPENAI_API_KEY` is configured, the OpenAI model writes a grounded natural-language answer; otherwise a local fallback still returns ranked products.

Example questions:
- `Recommend low-sugar vegetarian snacks under ₹100`
- `Suggest higher-protein options and avoid peanuts`
- `Which products fit a balanced diet?`

Limitations: recommendations depend on the coverage and quality of the included catalogue. The Random Forest is trained on synthetic preference labels, not expert-reviewed suitability labels. Check product packaging, allergen statements and nutrition basis.


## TensorFlow/Keras deep-learning model
The app includes an optional multilayer perceptron (MLP) for nutrition-based preference scoring:
- Input features: sugar, protein, fiber, sodium, calories, fat, carbohydrates, selected goals, vegetarian and vegan preferences.
- Network: Dense(32, ReLU) → Dropout(0.10) → Dense(16, ReLU) → Dense(1, sigmoid).
- The final recommendation blends the transparent rule score, Random Forest estimate and MLP estimate when each is available.
- The website's “AI model status & evaluation” section shows whether the MLP is active.

Install dependencies with `python -m pip install -r requirements.txt`. TensorFlow may take a few minutes to download. The app still runs if TensorFlow cannot be imported, but the MLP score is then unavailable and the interface reports that status.

**Academic limitation:** the supplied product CSV has no expert suitability labels. Both the Random Forest and MLP are demonstration models trained against synthetic labels created from the project's rules. Their metrics measure agreement with those synthetic labels, not medical or real-world recommendation accuracy. For a stronger model, collect human/expert-reviewed labels and evaluate on a held-out product set.


## Compare scanned products or dataset products
The comparison page supports both workflows: (1) upload and OCR-scan two physical product labels, or (2) search the included `data/packaged_foods_india.csv` catalogue and load any two products directly into Product A and Product B. You can also load one dataset product and scan the other. Dataset selection uses `/api/products`; values are copied into the same editable comparison form. Always verify the nutrition basis before comparing.


## Combined image dataset and multi-product comparison

This combined project includes the Indian Packaged Food Image Dataset Kit under
`Indian_Packaged_Food_Image_Dataset/` plus the existing SmartChoice website.

The **Multi-product scan & compare** section supports:
- OCR scanning of multiple selected product label images in one batch.
- Selecting multiple products from the 852-row packaged-food CSV.
- Selecting products with downloaded images from the image-dataset manifest (available after running the image downloader).
- Combining image-scanned and dataset-selected products in one list, editing extracted nutrition values, removing products, and comparing 2–100 products at once.
- Ranked output, nutrition comparison table, rule score, Random Forest/MLP score when available, XAI explanation, and optional Generative AI explanation.

### Image dataset downloader
Run from `SmartChoice_Scan_Compare_AI/Indian_Packaged_Food_Image_Dataset/`:
```powershell
py download_images.py --limit 100 --delay 7
```
Then process all rows:
```powershell
py download_images.py --limit 852 --delay 7
```
The downloader saves progress to `metadata/image_manifest.csv` and stores downloaded images under `images/`. The website shows downloaded image-dataset products that have existing image files.

### Important
- This is an OCR-assisted comparison system, not a guarantee that OCR values are correct.
- Compare products only after confirming the nutrition basis is consistent (e.g., per 100 g versus per serving).
- The CSV does not provide expert suitability labels; model metrics are against synthetic demonstration labels.
- Review product matches and image/data licensing before redistributing downloaded images.


## Front/back Indian product image dataset (852 source products)
A second dataset kit is included at `Indian_Packaged_Food_Image_Dataset/Front_Back_Matching_Dataset/`.
It contains the same 852-row source CSV, a row-aligned image manifest, and `download_dataset.py`.
The app's image-dataset selector can read images from both the original image kit and this new front/back kit after images have been downloaded.

In PowerShell, from `Indian_Packaged_Food_Image_Dataset\Front_Back_Matching_Dataset`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --default-timeout=120 --retries=10 -r requirements.txt
.\.venv\Scripts\python.exe download_dataset.py --limit 50 --delay 3
```

Review the downloaded matches, then process the full list with `--limit 852`. Front/back photos are not guaranteed for every product. Where no separate back photo exists, the manifest may use an ingredients-label or nutrition-label image as a back-label fallback. Check `metadata/image_manifest.csv` and visually verify all matches before use.
