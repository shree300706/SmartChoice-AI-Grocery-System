import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"
)
import os, re, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rag import LocalRAG, explain

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "packaged_foods_india.csv"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# OCR patterns for common nutrition-panel labels. Always manually verify results.
PATTERNS = {
 "sugar": r"(?:total\s+)?sugars?\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gm|grams?)?",
 "protein": r"proteins?\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gm|grams?)?",
 "fiber": r"(?:dietary\s+)?fib(?:re|er)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gm|grams?)?",
 "sodium": r"sodium\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:mg|milligrams?)?",
 "calories": r"(?:energy|calories|calorie)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:kcal|cal)?",
 "fat": r"(?:total\s+)?fat\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gm|grams?)?",
 "carbs": r"(?:total\s+)?carbohydrates?\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gm|grams?)?",
}
def parse_label(text):
    found = {}
    normalized = text.replace(",", ".")
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, normalized, re.I)
        if m:
            try: found[key] = float(m.group(1))
            except ValueError: pass
    m = re.search(r"(?:ingredients?|contains)\s*[:\-]?\s*(.{8,500}?)(?:\n|allergen|nutrition|per\s+100|per\s+serving|$)", text, re.I|re.S)
    if m: found["ingredients"] = re.sub(r"\s+", " ", m.group(1)).strip(" :-\n\t")
    return found

def run_ocr(path):
    im = Image.open(path).convert("RGB")
    im = ImageOps.grayscale(im)
    im = ImageEnhance.Contrast(im).enhance(1.7)
    im = im.filter(ImageFilter.SHARPEN)
    raw = pytesseract.image_to_string(im, config="--psm 6")
    return {"raw_text": raw, "nutrition": parse_label(raw)}

def nval(v):
    try:
        if v is None or str(v).strip() == "": return np.nan
        return float(v)
    except (ValueError, TypeError): return np.nan

GOALS = ["reduce_sugar", "higher_protein", "higher_fiber", "lower_sodium", "balanced"]
NUTRIENTS = ["sugar", "protein", "fiber", "sodium", "calories", "fat", "carbs"]
def rule_score(p, profile):
    score = 60.0
    goals = profile.get("goals", [])
    sugar, protein, fiber, sodium = [nval(p.get(k)) for k in ["sugar","protein","fiber","sodium"]]
    if "reduce_sugar" in goals and np.isfinite(sugar): score += float(np.clip(8-sugar,-18,8))*2
    if "higher_protein" in goals and np.isfinite(protein): score += min(12, protein*.8)
    if "higher_fiber" in goals and np.isfinite(fiber): score += min(12, fiber*1.2)
    if "lower_sodium" in goals and np.isfinite(sodium): score += float(np.clip((250-sodium)/35,-12,8))
    if "balanced" in goals:
        if np.isfinite(sugar): score += float(np.clip(4-sugar*.3,-8,4))
        if np.isfinite(fiber): score += min(5,fiber*.4)
    return float(np.clip(score,0,100))


def explainable_score(p, profile):
    """Explain the transparent heuristic score using per-feature contributions."""
    items = [{"feature": "Starting score", "value": 60.0,
              "reason": "Every product starts from the same neutral baseline."}]
    goals = profile.get("goals", [])
    sugar, protein, fiber, sodium = [nval(p.get(k)) for k in ["sugar", "protein", "fiber", "sodium"]]
    if "reduce_sugar" in goals:
        delta = float(np.clip(8 - sugar, -18, 8)) * 2 if np.isfinite(sugar) else 0.0
        reason = f"Sugar is {sugar:g}; this goal changes the rule score by {delta:+.1f} points." if np.isfinite(sugar) else "Sugar was not detected."
        items.append({"feature":"Sugar", "value":round(delta,2), "reason":reason})
    if "higher_protein" in goals:
        delta = min(12, protein * .8) if np.isfinite(protein) else 0.0
        reason = f"Detected protein: {protein:g}." if np.isfinite(protein) else "Protein was not detected."
        items.append({"feature":"Protein", "value":round(float(delta),2), "reason":reason})
    if "higher_fiber" in goals:
        delta = min(12, fiber * 1.2) if np.isfinite(fiber) else 0.0
        reason = f"Detected fiber: {fiber:g}." if np.isfinite(fiber) else "Fiber was not detected."
        items.append({"feature":"Fiber", "value":round(float(delta),2), "reason":reason})
    if "lower_sodium" in goals:
        delta = float(np.clip((250 - sodium) / 35, -12, 8)) if np.isfinite(sodium) else 0.0
        reason = f"Detected sodium: {sodium:g}." if np.isfinite(sodium) else "Sodium was not detected."
        items.append({"feature":"Sodium", "value":round(delta,2), "reason":reason})
    if "balanced" in goals:
        if np.isfinite(sugar):
            delta = float(np.clip(4 - sugar * .3, -8, 4))
            items.append({"feature":"Balanced goal — sugar", "value":round(delta,2),
                          "reason":f"Sugar contributes {delta:+.1f} points in the balanced-goal heuristic."})
        if np.isfinite(fiber):
            delta = min(5, fiber * .4)
            items.append({"feature":"Balanced goal — fiber", "value":round(float(delta),2),
                          "reason":f"Fiber contributes {delta:+.1f} points in the balanced-goal heuristic."})
    raw_total = sum(float(item["value"]) for item in items)
    return {"rule_score":round(float(np.clip(raw_total, 0, 100)),1),
            "unclipped_total":round(raw_total,2),
            "contributions":items,
            "interpretation":"Positive values increase the transparent rule score; negative values decrease it. This explains the heuristic component, not a medical assessment."}

def model_feature_importance():
    if not MODEL:
        return []
    names = NUTRIENTS + GOALS + ["vegetarian preference", "vegan preference"]
    values = MODEL["model"].feature_importances_
    return [{"feature":name, "importance":round(float(value),4)}
            for name,value in sorted(zip(names, values), key=lambda pair:pair[1], reverse=True)]

def warnings_for(p, profile):
    out=[]
    text=(str(p.get("name",""))+" "+str(p.get("ingredients",""))).casefold()
    for term in [x.strip() for x in str(profile.get("avoid","")).split(",") if x.strip()]:
        if term.casefold() in text: out.append("Possible avoided ingredient: "+term)
    if profile.get("vegetarian") and any(x in text for x in ["gelatin","chicken","beef","fish oil","pork","meat"]):
        out.append("Possible non-vegetarian ingredient; check the package")
    if profile.get("vegan") and any(x in text for x in ["milk","milk solids","whey","casein","honey","egg","gelatin"]):
        out.append("Possible non-vegan ingredient; check the package")
    for field, key, label in [("max_sugar","sugar","Sugar"),("max_sodium","sodium","Sodium"),("budget","price","Price")]:
        limit=nval(profile.get(field)); val=nval(p.get(key))
        if np.isfinite(limit) and np.isfinite(val) and val>limit:
            out.append(f"{label} exceeds your chosen limit ({limit:g})")
    return out

# Prepare the supplied packaged-food data and train a demonstration preference model.
def load_dataset():
    df=pd.read_csv(DATA).fillna("")
    aliases={"name":["Item name","item_name","product_name","name"],"brand":["Brand_Name","brand"],
             "category":["Category","category"],"ingredients":["Ingredients","ingredients"],
             "sugar":["Sugar_g","sugar_g"],"protein":["Proteins_g","Protein_g","protein_g"],
             "fiber":["Dietary_Fiber_g","Fiber_g"],"sodium":["Sodium_mg","sodium_mg"],
             "calories":["Calories_kcal","calories"],"fat":["Total_Fat_g","fat"],
             "carbs":["Carbohydrates_g","carbs"],"price":["Price_INR","price"]}
    out=pd.DataFrame()
    for key, options in aliases.items():
        col=next((c for c in options if c in df.columns),None)
        if col is None: out[key]="" if key in ["name","brand","category","ingredients"] else np.nan
        elif key in ["name","brand","category","ingredients"]: out[key]=df[col].astype(str)
        else: out[key]=pd.to_numeric(df[col].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)",expand=False),errors="coerce")
    return out

PRODUCTS=load_dataset()
def train_demo_model():
    if len(PRODUCTS)<5: return None, {"status":"Not enough dataset rows; rule score only"}
    rng=np.random.default_rng(42); X=[]; y=[]
    pools=[["reduce_sugar"],["higher_protein"],["higher_fiber"],["lower_sodium"],["balanced"],["reduce_sugar","higher_fiber"]]
    sample=PRODUCTS.sample(n=min(3500,max(500,len(PRODUCTS)*3)),replace=True,random_state=42)
    for _,row in sample.iterrows():
        p=row.to_dict(); profile={"goals":pools[int(rng.integers(len(pools)))]}
        vals=[nval(p.get(k)) for k in NUTRIENTS]+[float(g in profile["goals"]) for g in GOALS]+[0.0,0.0]
        X.append(vals); y.append(rule_score(p,profile)+rng.normal(0,1))
    X=np.array(X,float); med=np.nanmedian(X,axis=0); med[~np.isfinite(med)]=0; X=np.where(np.isnan(X),med,X); y=np.array(y)
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
    model=RandomForestRegressor(n_estimators=80,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(xt,yt)
    pred=model.predict(xv)
    return {"model":model,"medians":med},{"status":"RandomForest demo model trained","training_rows":len(xt),"test_rows":len(xv),
      "MAE":round(float(mean_absolute_error(yv,pred)),3),"RMSE":round(float(np.sqrt(mean_squared_error(yv,pred))),3),
      "R2":round(float(r2_score(yv,pred)),3),"note":"Metrics are against synthetic rule-generated labels, not expert-rated suitability."}
MODEL, METRICS=train_demo_model()

# Optional TensorFlow/Keras deep-learning model. Targets are generated from
# transparent preference rules because the supplied CSV has no expert labels.
def train_deep_learning_model():
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return None, {"status":"Deep learning unavailable; install tensorflow-cpu to enable it", "detail":str(exc)[:180]}
    if len(PRODUCTS) < 5:
        return None, {"status":"Not enough product rows for the demonstration MLP"}
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    goal_pools = [[], ["reduce_sugar"], ["higher_protein"], ["higher_fiber"],
                  ["lower_sodium"], ["balanced"], ["reduce_sugar","higher_fiber"],
                  ["higher_protein","lower_sodium"]]
    sample = PRODUCTS.sample(n=1800, replace=True, random_state=42)
    rows, targets = [], []
    for _, row in sample.iterrows():
        p = row.to_dict()
        goals = goal_pools[int(rng.integers(len(goal_pools)))]
        vegetarian = float(rng.integers(0, 2))
        vegan = float(rng.integers(0, 2)) if not vegetarian else 0.0
        vals = [nval(p.get(k)) for k in NUTRIENTS]
        vals += [float(g in goals) for g in GOALS] + [vegetarian, vegan]
        rows.append(vals)
        targets.append(float(np.clip(rule_score(p, {"goals":goals}), 0, 100)) / 100.0)
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0
    X = np.where(np.isfinite(X), X, med).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    xt, xv, yt, yv = train_test_split(X, y, test_size=.2, random_state=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.fit(xt, yt, validation_split=.15, epochs=18, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    pred = model.predict(xv, verbose=0).reshape(-1)
    return {"model":model, "scaler":scaler, "medians":med}, {
        "status":"TensorFlow/Keras MLP trained (demonstration)",
        "architecture":"Input → Dense(32, ReLU) → Dropout(0.10) → Dense(16, ReLU) → Sigmoid output",
        "training_rows":int(len(xt)), "test_rows":int(len(xv)),
        "MAE":round(float(mean_absolute_error(yv*100, pred*100)),3),
        "RMSE":round(float(np.sqrt(mean_squared_error(yv*100, pred*100))),3),
        "note":"Metrics compare against synthetic rule-generated preference labels, not expert or clinical suitability."
    }

DL_MODEL, DL_METRICS = train_deep_learning_model()
METRICS["deep_learning"] = DL_METRICS
RAG = LocalRAG()

def recommend(p, profile):
    r=rule_score(p,profile); ml=None
    if MODEL:
        vals=[nval(p.get(k)) for k in NUTRIENTS]+[float(g in profile.get("goals",[])) for g in GOALS]+[float(bool(profile.get("vegetarian"))),float(bool(profile.get("vegan")))]
        vals=np.where(np.isnan(np.array(vals,float)),MODEL["medians"],np.array(vals,float)).reshape(1,-1)
        try: ml=float(np.clip(MODEL["model"].predict(vals)[0],0,100))
        except Exception: ml=None
    dl = None
    if DL_MODEL:
        dl_vals = [nval(p.get(k)) for k in NUTRIENTS]
        dl_vals += [float(g in profile.get("goals", [])) for g in GOALS]
        dl_vals += [float(bool(profile.get("vegetarian"))), float(bool(profile.get("vegan")))]
        dl_arr = np.asarray(dl_vals, dtype=float)
        dl_arr = np.where(np.isfinite(dl_arr), dl_arr, DL_MODEL["medians"]).reshape(1, -1)
        try:
            dl = float(np.clip(DL_MODEL["model"].predict(DL_MODEL["scaler"].transform(dl_arr).astype(np.float32), verbose=0)[0][0] * 100, 0, 100))
        except Exception:
            dl = None
    available = [(0.50, r), (0.25, ml), (0.25, dl)]
    available = [(weight, value) for weight, value in available if value is not None]
    score = sum(weight * value for weight, value in available) / sum(weight for weight, value in available)
    warn=warnings_for(p,profile)
    match="Needs attention" if warn and score>=50 else ("Poor match" if score<50 else ("Good match" if score>=72 else "Needs attention"))
    reasons=[]
    for g,msg in [("reduce_sugar","Sugar considered for your reduce-sugar goal."),("higher_protein","Protein considered for your higher-protein goal."),("higher_fiber","Fiber considered for your higher-fiber goal."),("lower_sodium","Sodium considered for your lower-sodium goal.")]:
        if g in profile.get("goals",[]): reasons.append(msg)
    if not reasons: reasons.append("Score reflects selected goals and available nutrition values.")
    return {"score":round(float(score),1),"rule_score":round(r,1),"ml_score":round(ml,1) if ml is not None else None,
            "deep_learning_score":round(dl,1) if dl is not None else None,"match":match,"warnings":warn,"reasons":reasons}

def infer_chat_profile(message, base_profile=None):
    """Combine the user's saved form preferences with lightweight chat intent detection."""
    profile = dict(base_profile or {})
    text = str(message).casefold()
    goals = list(profile.get("goals", []))
    cues = {
        "reduce_sugar": ["low sugar", "less sugar", "reduce sugar", "diabetes", "sugar-free", "sugar free"],
        "higher_protein": ["high protein", "more protein", "protein-rich", "protein rich"],
        "higher_fiber": ["high fiber", "high fibre", "more fiber", "more fibre"],
        "lower_sodium": ["low sodium", "less salt", "low salt", "lower sodium"],
        "balanced": ["balanced", "healthier", "healthy snack", "healthy option"],
    }
    for goal, phrases in cues.items():
        if any(phrase in text for phrase in phrases) and goal not in goals:
            goals.append(goal)
    profile["goals"] = goals or ["balanced"]
    budget_match = re.search(r"(?:under|below|less than|budget(?: of)?|within)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)|₹\s*(\d+(?:\.\d+)?)", text)
    if budget_match:
        profile["budget"] = budget_match.group(1) or budget_match.group(2)
    if "vegan" in text:
        profile["vegan"] = True
        profile["vegetarian"] = False
    elif "vegetarian" in text:
        profile["vegetarian"] = True
    match = re.search(r"(?:avoid|allergic to|allergy to|can't eat|cannot eat|without)\s+([a-z ,/-]+)", text)
    if match:
        terms = [x.strip(" .?!") for x in re.split(r",| and |/", match.group(1)) if x.strip(" .?!")]
        existing = [x.strip() for x in str(profile.get("avoid", "")).split(",") if x.strip()]
        profile["avoid"] = ", ".join(dict.fromkeys(existing + terms))
    return profile

def catalog_recommendations(profile, limit=5, message=""):
    """Rank the provided catalogue using the same transparent preference model."""
    candidates = []
    text = str(message).casefold()
    category_terms = {"snack":"snack", "biscuit":"biscuit", "cookie":"biscuit", "juice":"juice", "chocolate":"chocolate", "sweet":"sweet", "ice cream":"ice", "dairy":"dairy", "sauce":"sauce", "instant food":"instant food", "wafer":"wafer", "cake":"cake", "spread":"spread"}
    wanted_categories = [term for term in category_terms if term in text]
    for _, row in PRODUCTS.iterrows():
        p = row.to_dict()
        if wanted_categories and not any(category_terms[term] in str(p.get("category", "")).casefold() or term in str(p.get("name", "")).casefold() for term in wanted_categories):
            continue
        if not any(np.isfinite(nval(p.get(k))) for k in ["sugar", "protein", "fiber", "sodium", "calories"]):
            continue
        result = recommend(p, profile)
        candidates.append({**p, **result, "explanation": explainable_score(p, profile)})
    candidates.sort(key=lambda p: (bool(p["warnings"]), -p["score"]))
    return candidates[:limit]

def chat_answer(message, profile, candidates, retrieved):
    """Ground the optional generative response in catalog rows and local retrieved guidance."""
    key = os.getenv("OPENAI_API_KEY")
    catalog = [{k:p.get(k) for k in ["name", "brand", "category", "ingredients", "sugar", "protein", "fiber", "sodium", "price", "score", "match", "warnings", "reasons"]} for p in candidates]
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            context = "\n".join(f"- {x['topic']}: {x['guidance']} (source: {x['source']})" for x in retrieved)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2,
                messages=[
                    {"role":"system","content":"You are SmartChoice, a packaged-food recommendation assistant. Recommend only products in the supplied catalog candidates. Ground all nutrition statements in supplied values; say when values are missing. Explain trade-offs simply. Treat scores as preference heuristics, not clinical facts. Never claim a product is allergy-safe. Tell users to verify the physical package, allergen statement, OCR values and per-serving/per-100g basis. Do not invent products, prices, nutrition values or sources."},
                    {"role":"user","content":f"User message: {message}\nProfile: {profile}\nCatalog candidates (source dataset, may have missing values): {catalog}\nRetrieved guidance: {context}\nAnswer helpfully and give up to 3 recommendations, with reasons and cautions."}
                ])
            return {"mode":"Generative AI + catalogue grounding + RAG", "generated":True, "text":response.choices[0].message.content}
        except Exception:
            pass
    if not candidates:
        return {"mode":"Local recommendation fallback", "generated":False, "text":"I couldn't find catalogue products with enough nutrition data for this request. Try relaxing the preferences or scan a product label to compare it."}
    lines = ["Here are catalogue matches ranked for your selected goals (scores are preference estimates, not health ratings):"]
    for i, p in enumerate(candidates[:3], 1):
        nutrients = []
        for key, label, unit in [("sugar","sugar","g"),("protein","protein","g"),("fiber","fiber","g"),("sodium","sodium","mg"),("price","price","INR")]:
            val = nval(p.get(key))
            if np.isfinite(val): nutrients.append(f"{label} {val:g} {unit}")
        lines.append(f"{i}. {p.get('name') or 'Unnamed product'} — {p['score']}/100, {p['match']}. " + "; ".join(nutrients) + ".")
        if p.get("warnings"):
            lines.append("   Caution: " + "; ".join(p["warnings"]) + ". Verify the package and allergen statement.")
        elif p.get("reasons"):
            lines.append("   Why: " + " ".join(p["reasons"]))
    if retrieved:
        lines.append("General guidance: " + " ".join(x["guidance"] for x in retrieved[:2]))
    lines.append("Check the label yourself; this tool cannot guarantee allergy safety or provide medical advice.")
    return {"mode":"Local catalogue recommender + RAG fallback", "generated":False, "text":"\n".join(lines)}

@app.route("/")
def home():
    return render_template("index.html",product_count=len(PRODUCTS),metrics=METRICS)

@app.route("/api/products", methods=["GET"])
def list_products():
    """Return searchable product choices from the included packaged-food dataset."""
    query = str(request.args.get("q", "")).strip().casefold()
    category = str(request.args.get("category", "")).strip().casefold()
    limit = min(max(int(request.args.get("limit", 1000)), 1), 2000)
    rows = []
    for idx, row in PRODUCTS.iterrows():
        p = row.to_dict()
        name = str(p.get("name", "") or "").strip()
        brand = str(p.get("brand", "") or "").strip()
        cat = str(p.get("category", "") or "").strip()
        if query and query not in f"{name} {brand} {cat}".casefold():
            continue
        if category and category not in cat.casefold():
            continue
        rows.append({"id": int(idx), "name": name or f"Product {idx+1}", "brand": brand,
                     "category": cat, "sugar": _json_num(p.get("sugar")),
                     "protein": _json_num(p.get("protein")), "fiber": _json_num(p.get("fiber")),
                     "sodium": _json_num(p.get("sodium")), "calories": _json_num(p.get("calories")),
                     "fat": _json_num(p.get("fat")), "carbs": _json_num(p.get("carbs")),
                     "price": _json_num(p.get("price")), "ingredients": str(p.get("ingredients", "") or "")})
        if len(rows) >= limit:
            break
    return jsonify({"products": rows, "count": len(rows), "dataset_count": int(len(PRODUCTS))})


@app.route("/api/ocr-batch", methods=["POST"])
def ocr_batch():
    """OCR multiple uploaded product images in one request."""
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error":"Choose one or more product images."}), 400
    if len(files) > 100:
        return jsonify({"error":"Upload up to 100 images per batch."}), 400
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            results.append({"name": f.filename, "error": "Unsupported image type; use JPG, PNG, WEBP, BMP or TIFF."})
            continue
        path = UPLOADS / (uuid.uuid4().hex + ext)
        try:
            f.save(path)
            result = run_ocr(path)
            nutrition = result.get("nutrition", {})
            item = {"name": Path(f.filename).stem, "filename": f.filename,
                    "ingredients": nutrition.get("ingredients", ""),
                    "sugar": nutrition.get("sugar"), "protein": nutrition.get("protein"),
                    "fiber": nutrition.get("fiber"), "sodium": nutrition.get("sodium"),
                    "calories": nutrition.get("calories"), "fat": nutrition.get("fat"),
                    "carbs": nutrition.get("carbs"), "price": None,
                    "raw_text": result.get("raw_text", "")}
            results.append(item)
        except Exception as exc:
            results.append({"name": f.filename, "filename": f.filename,
                            "error": f"OCR failed: {exc}. Check Tesseract OCR installation and PATH."})
        finally:
            path.unlink(missing_ok=True)
    return jsonify({"results": results, "count": len(results)})

@app.route("/api/image-products", methods=["GET"])
def image_products():
    """List products with downloaded images from either included image dataset."""
    image_root = BASE / "Indian_Packaged_Food_Image_Dataset"
    manifest_sources = [
        (image_root / "metadata" / "image_manifest.csv", ""),
        (image_root / "Front_Back_Matching_Dataset" / "metadata" / "image_manifest.csv", "Front_Back_Matching_Dataset/"),
    ]
    rows = []
    for manifest, prefix in manifest_sources:
        if not manifest.exists():
            continue
        try:
            df = pd.read_csv(manifest).fillna("")
        except Exception:
            continue
        for idx, row in df.iterrows():
            front = str(row.get("front_image_path", "")).strip()
            back = str(row.get("back_image_path", "")).strip()
            nutrition_img = str(row.get("nutrition_image_path", "")).strip()
            ingredients_img = str(row.get("ingredients_image_path", "")).strip()
            candidate_images = [front, back, nutrition_img, ingredients_img]
            existing = next((p for p in candidate_images if p and (manifest.parent.parent / p).is_file()), "")
            if not existing:
                continue
            def number(col):
                try:
                    v = float(row.get(col, ""))
                    return v if np.isfinite(v) else None
                except (TypeError, ValueError):
                    return None
            product_name = str(row.get("Item name", "") or row.get("product_name", "") or
                               row.get("matched_off_product_name", "") or f"Image product {idx+1}")
            brand = str(row.get("Brand_Name", "") or row.get("brand", "") or row.get("matched_brand", ""))
            def image_url(path):
                return ("/image-dataset/" + prefix + path.replace("\\", "/")) if path and (manifest.parent.parent / path).is_file() else None
            rows.append({
                "id": f"img-{len(rows)}", "name": product_name,
                "brand": brand,
                "category": str(row.get("Category", "") or row.get("category", "")),
                "ingredients": str(row.get("Ingredients", "")),
                "sugar": number("Sugar_g"), "protein": number("Proteins_g"),
                "fiber": number("Dietary_Fiber_g"), "sodium": number("Sodium_mg"),
                "calories": number("Calories_kcal"), "fat": number("Total_Fat_g"),
                "carbs": number("Carbohydrates_g"), "price": number("Price_INR"),
                "image_url": image_url(front) or image_url(existing),
                "back_image_url": image_url(back),
                "nutrition_image_url": image_url(nutrition_img),
                "ingredients_image_url": image_url(ingredients_img),
                "source": "front/back image dataset" if prefix else "image dataset"
            })
    return jsonify({"products": rows, "count": len(rows),
                    "message": "Image records appear after images have been downloaded into the image dataset folders."})


@app.route("/image-dataset/<path:filename>")
def serve_image_dataset(filename):
    return send_from_directory(BASE / "Indian_Packaged_Food_Image_Dataset", filename)

def _json_num(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

@app.route("/api/ocr",methods=["POST"])
def ocr():
    if "image" not in request.files or not request.files["image"].filename: return jsonify({"error":"Choose an image first"}),400
    f=request.files["image"]; ext=Path(f.filename).suffix.lower()
    if ext not in [".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"]: return jsonify({"error":"Unsupported image type"}),400
    path=UPLOADS/(uuid.uuid4().hex+ext); f.save(path)
    try: return jsonify(run_ocr(path))
    except Exception as e: return jsonify({"error":f"OCR failed: {e}. Install Tesseract OCR and ensure it is on PATH."}),500
    finally: path.unlink(missing_ok=True)

@app.route("/api/compare",methods=["POST"])
def compare():
    data=request.get_json(force=True); products=data.get("products",[]); profile=data.get("profile",{})
    if not isinstance(products, list) or len(products) < 2: return jsonify({"error":"Add at least two products to compare"}),400
    if len(products) > 100: return jsonify({"error":"Compare up to 100 products at a time"}),400
    results=[]
    for p in products:
        result = recommend(p,profile)
        xai = explainable_score(p, profile)
        results.append({**p,**result,"explanation":xai})
    results.sort(key=lambda p:(bool(p["warnings"]),-p["score"]))
    summary=f"Highest preference score: {results[0].get('name','Product')} ({results[0]['score']}/100)."
    if results[0]["warnings"]: summary+=" It has warnings; do not treat it as safe without checking the package."
    query = str({"profile":profile, "products":[{k:p.get(k) for k in ["name","sugar","protein","fiber","sodium","ingredients","score","match","warnings","explanation"]} for p in results]})
    retrieved = RAG.retrieve(query, k=3)
    explanation = explain(query, retrieved)
    return jsonify({"results":results,"summary":summary,"metrics":METRICS,
                    "knowledge":retrieved,"explanation":explanation,"global_feature_importance":model_feature_importance(),
                    "note":"Preference aid only. Verify OCR and nutrition basis; this is not medical advice."})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error":"Enter a question or recommendation request."}), 400
    profile = infer_chat_profile(message, data.get("profile", {}))
    candidates = catalog_recommendations(profile, limit=5, message=message)
    retrieved = RAG.retrieve(message + " " + str(profile), k=3)
    answer = chat_answer(message, profile, candidates, retrieved)
    shown = [{k:p.get(k) for k in ["name","brand","category","sugar","protein","fiber","sodium","price","score","match","warnings","reasons"]} for p in candidates[:3]]
    return jsonify({**answer, "recommendations":shown, "profile_used":profile,
                    "note":"Catalogue recommendations are heuristic. Confirm nutrition values and allergen statements on the physical package."})

if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)
