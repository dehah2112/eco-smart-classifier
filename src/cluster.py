"""
cluster.py - Module 3 : Clustering non-supervisé
Eco-Smart Classifier
"""
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
 
warnings.filterwarnings("ignore")
 
DATA_RAW       = Path("data/raw/dataset_ProjetML_2026.csv")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR     = Path("models")
METRICS_DIR    = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
 
NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
 
 
def preparer_clustering():
    df = pd.read_csv(DATA_RAW)
 
    def clip_iqr(s, f=3.0):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(q1 - f * (q3 - q1), q3 + f * (q3 - q1))
 
    for col in NUM_COLS:
        df[col] = clip_iqr(df[col])
 
    knn = KNNImputer(n_neighbors=5)
    df[NUM_COLS] = knn.fit_transform(df[NUM_COLS])
    df["Densite"]        = df["Poids"] / (df["Volume"] + 1e-9)
    df["Cond_Rig_Ratio"] = df["Conductivite"] / (df["Rigidite"] + 1e-9)
    df = pd.get_dummies(df, columns=["Source"], drop_first=False, prefix="Source")
 
    exclude = ["Categorie", "Rapport_Collecte"]
    feat_cols = [c for c in df.columns if c not in exclude]
    return df, feat_cols
 
 
def choisir_k_optimal(X_scaled: np.ndarray, k_range=range(2, 10)) -> int:
    """Elbow + Silhouette + Davies-Bouldin + Calinski-Harabasz → vote."""
    sil, db, ch, inertias = [], [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, lbl))
        db.append(davies_bouldin_score(X_scaled, lbl))
        ch.append(calinski_harabasz_score(X_scaled, lbl))
 
    k_list = list(k_range)
    votes = [
        k_list[int(np.argmax(sil))],
        k_list[int(np.argmin(db))],
        k_list[int(np.argmax(ch))],
    ]
    k_opt = Counter(votes).most_common(1)[0][0]
    print(f"  Silhouette → k={k_list[np.argmax(sil)]} | "
          f"DB → k={k_list[np.argmin(db)]} | "
          f"CH → k={k_list[np.argmax(ch)]}")
    print(f"  k optimal retenu (vote) : {k_opt}")
    return k_opt
 
 
def run():
    print("=" * 55)
    print("MODULE 3 – Clustering")
    print("=" * 55)
 
    df, feat_cols = preparer_clustering()
    X = df[feat_cols].values
 
    scaler_clust = StandardScaler()
    X_scaled     = scaler_clust.fit_transform(X)
    print(f"Dataset clustering : {X_scaled.shape}")
 
    k_optimal = choisir_k_optimal(X_scaled)
 
    km_final  = KMeans(n_clusters=k_optimal, init="k-means++",
                       n_init=20, max_iter=500, random_state=SEED)
    labels    = km_final.fit_predict(X_scaled)
 
    sil_final = silhouette_score(X_scaled, labels)
    db_final  = davies_bouldin_score(X_scaled, labels)
    ch_final  = calinski_harabasz_score(X_scaled, labels)
 
    print(f"K-Means final (k={k_optimal})")
    print(f"  Silhouette     : {sil_final:.4f}")
    print(f"  Davies-Bouldin : {db_final:.4f}")
    print(f"  Calinski-Harab : {ch_final:.0f}")
 
    dist = Counter(labels.tolist())
    for c, n in sorted(dist.items()):
        print(f"  Cluster {c} : {n} pts ({n/len(labels)*100:.1f}%)")
 
    # PCA 2D
    pca_2d   = PCA(n_components=2, random_state=SEED)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
 
    # Sauvegarde
    df_result = df.copy()
    df_result["Cluster_KMeans"] = labels
    df_result.to_csv(DATA_PROCESSED / "dataset_with_clusters.csv", index=False)
 
    for fname, obj in [
        ("kmeans.pkl",       km_final),
        ("scaler_clust.pkl", scaler_clust),
        ("pca_2d.pkl",       pca_2d),
    ]:
        with open(MODELS_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  OK models/{fname}")
 
    metrics = {
        "k_optimal":       k_optimal,
        "silhouette":      round(sil_final, 4),
        "davies_bouldin":  round(db_final, 4),
        "calinski_harabasz": round(ch_final, 2),
    }
    with open(METRICS_DIR / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print("\nOK kmeans.pkl et dataset_with_clusters.csv sauvegardés")
 
 
if __name__ == "__main__":
    run()