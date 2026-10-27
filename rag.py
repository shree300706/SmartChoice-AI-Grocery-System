import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")

class LocalRAG:
    def __init__(self, path=KB_PATH):
        self.kb = pd.read_csv(path).fillna("")
        self.docs = self.kb[["topic", "guidance", "source"]].astype(str).agg(" ".join, axis=1).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idxs = scores.argsort()[::-1][:k]
        return [{"topic":str(self.kb.iloc[i]["topic"]),
                 "guidance":str(self.kb.iloc[i]["guidance"]),
                 "source":str(self.kb.iloc[i]["source"]),
                 "similarity":round(float(scores[i]),3)} for i in idxs if scores[i] > 0]

def explain(query, retrieved):
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            context = "\n".join(f"- {x['topic']}: {x['guidance']} (source: {x['source']})" for x in retrieved)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {"role":"system","content":"You explain SmartChoice packaged-food comparisons. Use only supplied product values, score contributions, warnings and retrieved knowledge. Distinguish label facts from heuristics and general guidance. Do not invent missing values or sources. Never claim medical or allergy safety. Remind users to verify OCR and serving basis."},
                    {"role":"user","content":f"Comparison details and profile:\\n{query}\\n\\nRetrieved knowledge:\\n{context}\\n\\nExplain which product ranks higher, the main nutrient trade-offs, ingredient/limit warnings, missing data to verify, and which retrieved source labels support general guidance. Do not treat the heuristic score as a health prediction."}
                ])
            return {"mode":"Generative AI + RAG (OpenAI)", "generated":True, "model":os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "text":response.choices[0].message.content}
        except Exception as exc:
            fallback_note = f"LLM call failed ({str(exc)[:120]}); using local RAG."
    else:
        fallback_note = "OPENAI_API_KEY is not set; using local RAG."
    text = " ".join(x["guidance"] for x in retrieved[:2])
    return {"mode":"Local RAG fallback (no generated LLM answer)", "generated":False, "text":text + " " + fallback_note}
