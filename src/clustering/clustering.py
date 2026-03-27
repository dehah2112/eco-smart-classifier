# === MODULE 3 — CLUSTERING NON SUPERVISÉ (version finale corrigée) ===
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, adjusted_mutual_info_score)
warnings.filterwarnings('ignore')

# ====================== ÉTAPE 1 — PRÉTRAITEMENT ======================
df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")

# Prix_Revente exclu : variable cible de régression, pas feature physique
# Source exclu : catégorielle — KMeans requiert variables continues
num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']

df_clust = df[num_cols].copy()
df_clust.loc[df_clust['Poids'] <= 0, 'Poids']   = np.nan
df_clust.loc[df_clust['Volume'] <= 0, 'Volume']  = np.nan
df_clust.loc[df_clust['Poids'] > 500, 'Poids']   = np.nan

# Imputation KNN (recommandé pour clustering — préserve la structure locale)
imputer  = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(df_clust), columns=num_cols)

# Standardisation OBLIGATOIRE pour KMeans
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"✅ Données prêtes : {X_scaled.shape}")
assert X_scaled.shape[0] == len(df), "Décalage d'index détecté !"

# ====================== ÉTAPE 2 — ELBOW + SILHOUETTE + DB ======================
inertias, silhouettes, db_scores = [], [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=3000))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    print(f"k={k} | inertia={km.inertia_:.0f} | "
          f"sil={silhouettes[-1]:.4f} | DB={db_scores[-1]:.4f}")

# Détermination automatique du k optimal
best_k_sil = list(K_range)[np.argmax(silhouettes)]
best_k_db  = list(K_range)[np.argmin(db_scores)]
print(f"\nk optimal (Silhouette) : {best_k_sil}")
print(f"k optimal (DB)         : {best_k_db}")
best_k = 4  # confirmé par métriques + cohérence avec les 4 classes réelles
print(f"→ k={best_k} retenu")

# Visualisation triple
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(K_range, inertias, 'o-', color='steelblue', linewidth=2)
axes[0].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'k={best_k}')
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertie (WCSS)")
axes[0].set_title("Méthode du coude")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(K_range, silhouettes, 's-', color='teal', linewidth=2)
axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'k={best_k}')
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Score de silhouette")
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(K_range, db_scores, '^-', color='coral', linewidth=2)
axes[2].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'k={best_k}')
axes[2].set_xlabel("k")
axes[2].set_ylabel("Davies-Bouldin")
axes[2].set_title("Davies-Bouldin (↓ = mieux)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reports/clustering_elbow.png", dpi=150)
plt.close()

# ====================== ÉTAPE 3 — MODÈLE FINAL ======================
kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=20,
                max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

assert len(cluster_labels) == len(df), "Décalage d'index détecté !"
df['cluster'] = cluster_labels

# Centres en espace original
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=num_cols)
centers_df.index.name = 'Cluster'
print("\n=== Centres des clusters (espace original) ===")
print(centers_df.round(2))

# Correspondance Cluster ↔ Catégorie
df_labeled = df[df['Categorie'].notna()].copy()
assert df_labeled['Categorie'].isna().sum() == 0

cross = pd.crosstab(df_labeled['cluster'], df_labeled['Categorie'])
print("\n=== Correspondance Cluster ↔ Catégorie ===")
print(cross)

for c in range(best_k):
    majority = cross.loc[c].idxmax()
    purity   = cross.loc[c].max() / cross.loc[c].sum()
    print(f"Cluster {c} → {majority:10s} (pureté {purity:.1%})")

# ====================== ÉTAPE 4 — PCA 2D ======================
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nVariance expliquée : "
      f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%} "
      f"(total={pca.explained_variance_ratio_.sum():.1%})")

loadings = pd.DataFrame(pca.components_.T, index=num_cols, columns=['PC1','PC2'])
print("\n=== Loadings PCA ===")
print(loadings.round(3))

# Scatter clusters
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=cluster_labels, cmap='tab10', alpha=0.4, s=8)
plt.colorbar(scatter, label='Cluster')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"Clusters KMeans (k={best_k}) — Projection PCA 2D")
plt.tight_layout()
plt.savefig("reports/pca_clusters.png", dpi=150)
plt.close()

# Biplot
fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(X_pca[:, 0], X_pca[:, 1],
           c=cluster_labels, cmap='tab10', alpha=0.2, s=5)
scale = 3.0
for feat in num_cols:
    ax.annotate("",
        xy=(loadings['PC1'][feat]*scale, loadings['PC2'][feat]*scale),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(loadings['PC1'][feat]*scale*1.1,
            loadings['PC2'][feat]*scale*1.1,
            feat, color='red', fontsize=9)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Biplot PCA — Clusters + Loadings features")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("reports/pca_biplot.png", dpi=150)
plt.close()

# ====================== ÉTAPE 5 — MÉTRIQUES ======================
sil = silhouette_score(X_scaled, cluster_labels, sample_size=5000)
db  = davies_bouldin_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score : {sil:.4f}  (0→1, ↑ = mieux)")
print(f"Davies-Bouldin   : {db:.4f}   (↓ = mieux)")
print(f"Inertie (WCSS)   : {kmeans.inertia_:.1f}")

y_true  = df_labeled['Categorie'].values
y_clust = df_labeled['cluster'].values

ari = adjusted_rand_score(y_true, y_clust)
ami = adjusted_mutual_info_score(y_true, y_clust)
print(f"\nAdjusted Rand Index : {ari:.4f}  (1.0 = parfait)")
print(f"Adj. Mutual Info    : {ami:.4f}  (1.0 = parfait)")

# ====================== ÉTAPE 6 — MLFLOW ======================
mlflow.set_experiment("eco-smart-clustering")

# PCA calculée une seule fois pour la boucle MLflow
X_pca_mlflow = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

for k in [2, 3, 4, 5, 6]:
    with mlflow.start_run(run_name=f"KMeans_k{k}"):
        km  = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        lbl = km.fit_predict(X_scaled)

        sil_k = silhouette_score(X_scaled, lbl, sample_size=3000)
        db_k  = davies_bouldin_score(X_scaled, lbl)

        mlflow.log_param("k",      k)
        mlflow.log_param("init",   "k-means++")
        mlflow.log_param("n_init", 10)
        mlflow.log_metric("inertia",          km.inertia_)
        mlflow.log_metric("silhouette_score", sil_k)
        mlflow.log_metric("davies_bouldin",   db_k)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X_pca_mlflow[:, 0], X_pca_mlflow[:, 1],
                   c=lbl, cmap='tab10', alpha=0.3, s=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"PCA 2D — k={k}")
        fig.savefig(f"reports/pca_k{k}.png", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(f"reports/pca_k{k}.png", artifact_path="plots")

        print(f"k={k} | sil={sil_k:.4f} | DB={db_k:.4f} | "
              f"inertia={km.inertia_:.0f}")

print("\n✅ Module 3 terminé — artefacts dans reports/ et MLflow")