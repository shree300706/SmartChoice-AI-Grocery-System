# 🛒 SmartChoice: Food Decision Support System

> **A decision-support system that helps users make more personalized grocery choices by considering product nutrition, ingredients, dietary preferences, and individual goals.**

## 📌 Overview

Choosing between similar food products at a grocery store can be surprisingly difficult.

Users often have to read ingredient lists, compare nutritional values, consider dietary preferences, and decide which product is a better fit for their individual needs.

**SmartChoice** was developed to address this problem by bringing product information and user preferences together in one decision-support system.

The system can analyze packaged-food information, extract nutritional details from product-label images, compare products, apply personalized scoring, and provide explanations for its recommendations.

SmartChoice was developed as a **semester mini-project for the AIES course** by **Siddhi, Addya, and Shreyash**.

---

# 🎯 Problem Statement

When selecting packaged food products, consumers may need to consider multiple factors simultaneously:

- Nutritional values
- Ingredients
- Dietary preferences
- Personal goals
- Product category
- Budget
- Potential ingredient restrictions

Manually comparing all these factors can be time-consuming and difficult, especially when nutritional information is presented in different formats.

SmartChoice aims to simplify this process by providing a personalized food decision-support workflow.

---

# 💡 Our Approach

SmartChoice considers both sides of the decision:

```text
             USER
              │
              │ Preferences
              │ Dietary Goals
              │ Budget
              ▼
       ┌───────────────┐
       │   SmartChoice │
       └───────┬───────┘
               │
               │
PRODUCT ───────┤
               │
Nutrition      │
Ingredients    │
Category       │
               ▼
       Personalized Analysis
               │
       ┌───────┴────────┐
       ▼                ▼
 Recommendation     Explanation
       │                │
       └────────┬───────┘
                ▼
          User Decision
```

Instead of treating every user the same, SmartChoice attempts to make the recommendation dependent on the user's selected goals and preferences.

---

# ✨ Key Features

## 🥗 1. Personalized Food Recommendations

Users can specify goals such as:

- Reduce sugar
- Higher protein
- Higher fiber
- Lower sodium
- Balanced nutrition

SmartChoice evaluates product information against these preferences and generates a personalized score.

---

## 📷 2. Nutrition Label OCR

SmartChoice supports extracting nutritional information from food-product label images using **Tesseract OCR**.

The system attempts to identify information such as:

- Calories
- Protein
- Carbohydrates
- Fat
- Sugar
- Fiber
- Sodium
- Ingredients

### Workflow

```text
Food Label Image
       ↓
Image Preprocessing
       ↓
OCR using Tesseract
       ↓
Text Extraction
       ↓
Nutrition Information Parsing
       ↓
Product Analysis
```

This allows users to work with information directly from a product label rather than manually entering every value.

---

# 🔎 3. Product Comparison

SmartChoice allows users to compare multiple food products using their available nutritional and product information.

Products can be evaluated based on factors such as:

- Sugar
- Protein
- Fiber
- Sodium
- Calories
- Fat
- Carbohydrates
- Ingredients
- User-selected preferences

The system can then identify which product better matches the selected requirements.

---

# ⚠️ 4. Dietary and Ingredient Warnings

SmartChoice checks product information against selected dietary preferences and ingredient restrictions.

For example, the system can identify potentially relevant ingredients for preferences such as:

- Vegetarian
- Vegan
- Avoided ingredients
- Sugar limits
- Sodium limits
- Budget constraints

These checks are intended as **decision-support signals**, not guarantees of safety.

---

# 💡 5. Explainable Recommendations

SmartChoice does not only return a numerical score.

The recommendation system also provides an explanation of the factors that contributed to the score.

For example:

```text
Product Score: 78/100

✓ Lower sugar
✓ Good protein content
✓ Good fiber content
⚠ Relatively high sodium
```

This makes the recommendation easier to understand instead of treating the score as a black box.

---

# 🤖 6. AI / ML Components

SmartChoice explores multiple approaches for personalized recommendation.

### Rule-Based Scoring

A transparent scoring system evaluates product nutritional values against user-selected goals.

### Random Forest

A `RandomForestRegressor` is used as an experimental recommendation model.

### Neural Network

A TensorFlow/Keras-based multilayer neural network is also explored as part of the recommendation experiments.

### Important Note

The current ML experiments use **synthetic preference labels generated from the project's scoring logic** rather than expert-annotated food suitability ratings.

Therefore, the current ML models should be considered **experimental components**, not clinically validated prediction models.

---

# 📚 7. RAG-Based Knowledge Retrieval

SmartChoice includes a local retrieval system using a nutrition-related knowledge base.

The system uses:

- TF-IDF
- Cosine similarity
- Local knowledge-base retrieval

to identify relevant information for user queries.

### Workflow

```text
User Query
    ↓
TF-IDF Representation
    ↓
Similarity Search
    ↓
Relevant Knowledge Retrieved
    ↓
Context for Explanation
```

This allows the system to retrieve relevant information instead of relying only on hard-coded responses.

---

# 💬 8. Natural Language Interaction

SmartChoice includes a chatbot interface for food-related queries.

The system can identify information from natural-language requests such as:

```text
"Suggest a low-sugar vegetarian snack under ₹100."
```

The request can be interpreted into relevant factors such as:

```text
Goal       → Reduce sugar
Diet       → Vegetarian
Budget     → ₹100
Category   → Snack
```

These factors can then be used during product recommendation.

An optional OpenAI API integration can be used for generating natural-language explanations based on the available product and retrieved information.

---

# 📱 9. Experimental Android Prototype

As an extension of the main project, the team also explored an **Android-based interface for direct food-label scanning**.

The prototype was designed to:

- Capture or load product-label images
- Extract ingredient and nutritional information
- Populate product comparison fields
- Connect the scanning concept with the SmartChoice recommendation workflow

The Android prototype is currently **experimental and was not fully integrated into the final working system**.

This component represents a possible direction for future development.

---

# 🏗️ System Architecture

```text
                         USER
                          │
                          ▼
                  ┌───────────────┐
                  │  Web Interface │
                  └───────┬───────┘
                          │
                          ▼
                    Flask Backend
                       app.py
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
        OCR          Product Dataset      RAG
          │               │                │
          │               │                ▼
          │               │        Knowledge Base
          │               │                │
          └───────┬───────┘                │
                  ▼                        │
          Nutrition Processing             │
                  │                        │
                  ▼                        │
          Personalized Scoring ◄──────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Rules       RF        MLP
        │         │         │
        └─────────┼─────────┘
                  ▼
        Recommendation Score
                  │
                  ▼
          Explanation / Result
```

---

# 🔄 Recommendation Workflow

```text
1. User provides preferences
             ↓
2. Product information is loaded
             ↓
3. Product label may be processed using OCR
             ↓
4. Nutritional and ingredient information is extracted
             ↓
5. User preferences are compared with product attributes
             ↓
6. Rule-based score is calculated
             ↓
7. Experimental ML models provide additional scoring
             ↓
8. Scores are combined
             ↓
9. Potential warnings are generated
             ↓
10. Recommendation and explanation are shown
```

---

# 🛠️ Technology Stack

## Backend

- Python
- Flask

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- Random Forest
- TensorFlow
- Keras

## OCR / Image Processing

- Tesseract OCR
- Pytesseract
- Pillow

## Recommendation

- Rule-based scoring
- Random Forest regression
- Neural-network-based experimental scoring

## Retrieval / RAG

- TF-IDF
- Cosine Similarity
- Local knowledge base

## AI Integration

- OpenAI API (optional)

## Frontend

- HTML
- CSS
- JavaScript

## Mobile Prototype

- Android prototype for experimental direct label scanning

---

# 📂 Project Structure

```text
SmartChoice-AI-Grocery-System/
│
├── app.py
├── rag.py
├── requirements.txt
│
├── data/
│   ├── knowledge_base.csv
│   └── packaged_foods_india.csv
│
├── templates/
│   └── index.html
│
├── SmartChoice_Indian_Product_Front_Back_Dataset/
│   ├── README.md
│   ├── SOURCE_AND_LICENSE.md
│   ├── download_dataset.py
│   ├── images/
│   └── metadata/
│
├── COMBINED_DATASET_GUIDE.md
├── MULTI_PRODUCT_GUIDE.md
├── .gitignore
└── README.md
```

---

# 📊 Dataset

SmartChoice uses packaged-food product information containing product and nutritional attributes.

The main product dataset includes information such as:

- Product name
- Brand
- Category
- Ingredients
- Calories
- Protein
- Fiber
- Sugar
- Sodium
- Fat
- Carbohydrates
- Price

The project also includes supporting resources for an Indian packaged-food front/back image dataset used during experimentation with product-label processing.

Additional dataset information is available in:

- `COMBINED_DATASET_GUIDE.md`
- `MULTI_PRODUCT_GUIDE.md`
- `SmartChoice_Indian_Product_Front_Back_Dataset/`

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/shree300706/SmartChoice-AI-Grocery-System.git
```

```bash
cd SmartChoice-AI-Grocery-System
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Tesseract OCR

SmartChoice uses Tesseract OCR for extracting information from food-label images.

Install Tesseract OCR separately and ensure that its executable is correctly configured for your system.

## 5. Optional OpenAI API configuration

Some AI explanation functionality can use the OpenAI API.

If you use this functionality, configure your API key through an environment variable rather than placing the key directly in the source code.

**Never commit API keys or other secrets to GitHub.**

## 6. Run the application

```bash
python app.py
```

Open the local Flask address displayed in your terminal.

---

# 🖥️ Application Workflow

The main SmartChoice interface provides access to the food recommendation workflow.

A typical interaction can be:

```text
Select user preferences
        ↓
Search/select a product
        ↓
OR upload a food-label image
        ↓
Extract nutrition information
        ↓
Compare product with user goals
        ↓
Generate personalized score
        ↓
Show explanation and warnings
```

---

# ⚠️ Limitations

SmartChoice is an **academic/student project and decision-support system**, not a medical or clinical application.

The current system has several limitations:

### OCR Accuracy

OCR results can be affected by:

- Image quality
- Lighting
- Text orientation
- Font size
- Label design

Extracted values should therefore be verified against the original product label.

### Ingredient Detection

The current ingredient checks rely partly on keyword matching.

This means they **cannot guarantee that a product is safe for a particular allergy, medical condition, or dietary requirement**.

### ML Evaluation

The current ML experiments use synthetic preference labels generated from the project's scoring logic.

They are not trained or evaluated against expert-validated food suitability ratings.

### Dataset Coverage

Recommendations are dependent on the availability and quality of product information in the dataset.

### Health-Related Decisions

SmartChoice should not be used as a substitute for advice from qualified healthcare, nutrition, or diet professionals.

---

# 🔮 Future Scope

The project can be further improved in several directions.

### Recommendation System

- Improve recommendation accuracy
- Introduce better real-world evaluation
- Use higher-quality preference labels
- Improve personalization
- Explore more robust recommendation models

### OCR

- Improve label detection
- Improve OCR preprocessing
- Handle different packaging layouts
- Improve extraction of nutrition tables
- Add better validation of extracted values

### Ingredient Analysis

- Improve allergen detection
- Handle synonyms and ingredient variations
- Improve dietary classification
- Develop more robust ingredient parsing

### RAG

- Expand the knowledge base
- Improve retrieval quality
- Improve source-grounded explanations
- Add better context handling

### Mobile Application

- Complete the Android scanning workflow
- Improve direct camera-based label scanning
- Connect the Android application with the SmartChoice backend
- Enable real-time product analysis

### Deployment

- Improve cross-platform compatibility
- Containerize the application
- Deploy the backend
- Improve production configuration
- Add authentication and user profiles

---

# 🧪 Project Status

**Current Status: Working academic prototype**

The core SmartChoice web application provides:

- Personalized product scoring
- Product comparison
- Nutrition-label OCR
- Ingredient and dietary checks
- Experimental ML recommendation models
- RAG-based retrieval
- Natural-language interaction

The Android direct-scanning component remains experimental and is planned for further development.

---

# 👥 Team

This project was developed as a **semester mini-project for the AIES course**.

### Team Members

- **Siddhi Singh**
- **Shreyash Kewate**
- **Addya Nagle**

---

# 📌 Disclaimer

SmartChoice provides food-related information and personalized decision-support based on the data available to the system.

It is **not a medical diagnosis, medical recommendation, or substitute for professional dietary or healthcare advice**.

Always verify nutritional and ingredient information using the original product packaging and consult a qualified professional when making health-related dietary decisions.

---

# 🌱 Project Vision

SmartChoice started from a simple real-world question:

> **"Which product is actually a better choice for this particular person?"**

The project explores how AI, OCR, recommendation systems, and retrieval-based approaches can work together to make everyday grocery decisions more personalized and easier to understand.

The current version is a foundation, and the project will continue to evolve through improvements in recommendation quality, mobile integration, data coverage, and explainability.
