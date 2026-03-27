# === MODULE 4 — NLP (version finale corrigée) ===
import pandas as pd
import numpy as np
import mlflow
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from gensim.models import Word2Vec, FastText
warnings.filterwarnings('ignore')

# ====================== 1. PRÉTRAITEMENT NLP ======================
# 🔴 Section obligatoire — IA interdite selon la charte
nlp_model = spacy.load("fr_core_news_sm")

DOMAIN_STOPWORDS = {
    "déchet", "matériau", "collecte", "objet", "article",
    "type", "matière", "élément", "produit", "item", "lot"
}
ALL_STOPWORDS = SPACY_STOPWORDS.union(DOMAIN_STOPWORDS)

def preprocess_text(text: str) -> str:
    """Pipeline NLP : nettoyage → tokenisation → stopwords → lemmatisation"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc  = nlp_model(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in ALL_STOPWORDS
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return " ".join(tokens)

# Générer dataset_with_nlp.csv si pas encore fait
df_raw = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
df_raw['text_processed'] = df_raw['Rapport_Collecte'].fillna("").apply(preprocess_text)
df_raw.to_csv("data/processed/dataset_with_nlp.csv", index=False)
print(f"✅ Prétraitement NLP : {df_raw['text_processed'].str.len().describe()}")

# ====================== 2. CHARGEMENT ET ALIGNEMENT DES SPLITS ======================
df = pd.read_csv("data/processed/dataset_with_nlp.csv")
df_labeled = df[df['Categorie'].notna()].reset_index(drop=True)

# Indices sauvegardés depuis Module 1
train_idx = pd.read_csv('data/processed/train_idx.csv').squeeze().values
val_idx   = pd.read_csv('data/processed/val_idx.csv').squeeze().values
test_idx  = pd.read_csv('data/processed/test_idx.csv').squeeze().values

X_text = df_labeled['text_processed']
y      = df_labeled['Categorie']

X_tr  = X_text.loc[train_idx].reset_index(drop=True)
X_val = X_text.loc[val_idx].reset_index(drop=True)
X_te  = X_text.loc[test_idx].reset_index(drop=True)

y_tr  = y.loc[train_idx].reset_index(drop=True)
y_val = y.loc[val_idx].reset_index(drop=True)
y_te  = y.loc[test_idx].reset_index(drop=True)

le = joblib.load('models/label_encoder.pkl')
print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

# ====================== 3. VECTORISEURS SPARSE ======================
bow_vec = CountVectorizer(
    max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 1))
X_train_bow = bow_vec.fit_transform(X_tr)

tfidf_vec = TfidfVectorizer(
    max_features=8000, min_df=2, max_df=0.90,
    ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(X_tr)

# ====================== 4. WORD2VEC + FASTTEXT ======================
sentences_all = [t.split() for t in pd.concat([X_tr, X_val, X_te])]

w2v_model = Word2Vec(sentences=sentences_all, vector_size=100, window=5,
                     min_count=2, workers=4, sg=1, seed=42, epochs=10)
w2v_model.save("models/word2vec.model")

ft_model = FastText(sentences=sentences_all, vector_size=100, window=5,
                    min_count=2, workers=4, sg=1, seed=42, epochs=10)
ft_model.save("models/fasttext.model")

def vectorize_mean(texts, model, vector_size=100):
    """Mean pooling des vecteurs de mots"""
    vectors = []
    for text in texts:
        vecs = [model.wv[w] for w in text.split() if w in model.wv]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(vector_size))
    return np.array(vectors)

X_train_w2v = vectorize_mean(X_tr,  w2v_model)
X_val_w2v   = vectorize_mean(X_val, w2v_model)
X_test_w2v  = vectorize_mean(X_te,  w2v_model)

X_train_ft  = vectorize_mean(X_tr,  ft_model)
X_val_ft    = vectorize_mean(X_val, ft_model)
X_test_ft   = vectorize_mean(X_te,  ft_model)

# Sauvegarde vectoriseurs
joblib.dump(bow_vec,   'models/bow_vectorizer.pkl')
joblib.dump(tfidf_vec, 'models/tfidf_vectorizer.pkl')

# ====================== 5. DICTIONNAIRE VECTORISEURS ======================
# Format : (fitted_vec, X_train, type, X_val, X_test)
vectorizers = {
    "BoW":      (bow_vec,   X_train_bow,   "sparse",
                 bow_vec.transform(X_val),   bow_vec.transform(X_te)),
    "TF-IDF":   (tfidf_vec, X_train_tfidf, "sparse",
                 tfidf_vec.transform(X_val), tfidf_vec.transform(X_te)),
    "Word2Vec": (w2v_model, X_train_w2v,   "dense",
                 X_val_w2v,                 X_test_w2v),
    "FastText": (ft_model,  X_train_ft,    "dense",
                 X_val_ft,                  X_test_ft),
}

classifiers = {
    "MultinomialNB": lambda: MultinomialNB(alpha=0.1),
    "LinearSVC":     lambda: LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "LogReg":        lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "RandomForest":  lambda: RandomForestClassifier(n_estimators=100,
                                                     n_jobs=-1, random_state=42),
}

# ====================== 6. COMPARAISON VECTORISEURS × CLASSIFIEURS ======================
mlflow.set_experiment("eco-smart-nlp")

results      = {}
best_models  = {}

for vec_name, (vec, X_tr_vec, vec_type, X_v_vec, X_te_vec) in vectorizers.items():
    results[vec_name] = {}
    for clf_name, clf_factory in classifiers.items():
        # MultinomialNB incompatible avec vecteurs denses (valeurs négatives)
        if clf_name == "MultinomialNB" and vec_type == "dense":
            continue

        with mlflow.start_run(run_name=f"{vec_name}_{clf_name}"):
            clf = clf_factory()
            clf.fit(X_tr_vec, y_tr)

            y_pred = clf.predict(X_v_vec)
            f1  = f1_score(y_val, y_pred, average='macro')
            acc = accuracy_score(y_val, y_pred)

            results[vec_name][clf_name] = f1
            best_models[(vec_name, clf_name)] = clf

            mlflow.log_param("vectorizer", vec_name)
            mlflow.log_param("classifier", clf_name)
            mlflow.log_metric("val_f1_macro", f1)
            mlflow.log_metric("val_accuracy", acc)

            print(f"{vec_name:8s} + {clf_name:12s} "
                  f"→ F1={f1:.4f} | acc={acc:.4f}")

# ====================== 7. TABLEAU + HEATMAP ======================
results_df = pd.DataFrame(results).T
print("\n=== TABLEAU COMPARATIF F1-MACRO ===")
print(results_df.round(4))
results_df.to_csv("reports/nlp_comparison.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(results_df.astype(float), annot=True, fmt=".3f",
            cmap="YlGn", vmin=0.65)
plt.title("F1-macro — Vectorisation × Classifieur (validation)")
plt.ylabel("Vectorisation"); plt.xlabel("Classifieur")
plt.tight_layout()
plt.savefig("reports/nlp_comparison_heatmap.png", dpi=200)
plt.close()

# ====================== 8. ÉVALUATION FINALE SUR TEST SET ======================
best_combo = max(
    [(v, c, results[v].get(c, 0))
     for v in results for c in results[v]],
    key=lambda x: x[2]
)
best_vec_name, best_clf_name, best_val_f1 = best_combo
print(f"\n✅ Meilleure combinaison : {best_vec_name} + {best_clf_name} "
      f"(val F1={best_val_f1:.4f})")

best_clf  = best_models[(best_vec_name, best_clf_name)]
X_te_best = vectorizers[best_vec_name][4]  # X_test du meilleur vec
y_te_pred = best_clf.predict(X_te_best)

print("\n=== Évaluation FINALE NLP (test set) ===")
print(classification_report(y_te, y_te_pred,
      target_names=['Métal', 'Papier', 'Plastique', 'Verre']))

with mlflow.start_run(run_name=f"{best_vec_name}_{best_clf_name}_FINAL_TEST"):
    mlflow.log_param("vectorizer", best_vec_name)
    mlflow.log_param("classifier", best_clf_name)
    mlflow.log_metric("test_f1_macro",
                      f1_score(y_te, y_te_pred, average='macro'))
    mlflow.log_metric("test_accuracy",
                      accuracy_score(y_te, y_te_pred))

# Sauvegarde meilleur pipeline NLP
joblib.dump(best_clf, 'models/best_nlp_classifier.pkl')
print(f"✅ Modèle NLP sauvegardé → models/best_nlp_classifier.pkl")
print("\n✅ Module 4 terminé — artefacts dans reports/ et models/")