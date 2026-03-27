# ============================================================
# GUIDE DE DÉPLOIEMENT — Eco-Smart Classifier
# ============================================================

## Structure du projet attendue

eco-smart-classifier/
├── app_streamlit.py          ← Application Streamlit (ce fichier)
├── requirements.txt
├── models/
│   ├── best_multimodal_pipeline.pkl   ← Module 5
│   ├── best_regressor.pkl             ← Module 2
│   ├── label_encoder.pkl              ← Module 1
│   └── scaler.pkl                     ← Module 1
├── data/
│   └── raw/
│       └── dataset_ProjetML_2026.csv  ← Dataset brut
└── app/
    └── main.py                        ← API FastAPI (Module 6)


## Lancement local

# 1. Installer les dépendances
pip install -r requirements.txt
python -m spacy download fr_core_news_sm

# 2. S'assurer que les modèles sont dans models/
# (générés par les scripts des Modules 1-5)

# 3. Lancer l'application
streamlit run app_streamlit.py
# → http://localhost:8501


## Déploiement Streamlit Cloud (BONUS +2 pts)

# 1. Pusher le code sur GitHub
git add app_streamlit.py requirements.txt
git commit -m "feat: add streamlit app"
git push origin main

# 2. Sur https://share.streamlit.io :
#    - Connecter votre repo GitHub
#    - Main file path : app_streamlit.py
#    - Cliquer Deploy

# Note : les modèles .pkl doivent être dans le repo ou
# chargés depuis un stockage externe (HuggingFace Hub, GDrive)


## Déploiement Hugging Face Spaces (alternative BONUS)

# 1. Créer un Space sur huggingface.co/spaces
#    Type : Streamlit

# 2. Ajouter un fichier README.md avec metadata :
---
title: Eco-Smart Classifier
emoji: ♻️
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app_streamlit.py
pinned: false
---

# 3. Pusher le code
git remote add space https://huggingface.co/spaces/VOTRE_USER/eco-smart
git push space main


## Comportement sans modèles

L'application est conçue pour fonctionner en mode démonstration
si les fichiers .pkl ne sont pas disponibles :
- Onglet Dashboard : données synthétiques générées automatiquement
- Prédiction Manuelle : heuristiques physiques de démonstration
- Assistant NLP : correspondance par mots-clés

Les avertissements ⚠️ dans la sidebar indiquent les modèles manquants.


## Variables d'environnement (optionnel)

Créer un fichier .streamlit/secrets.toml pour la config :
[general]
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_PATH = "models/"
