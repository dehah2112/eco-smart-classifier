# PROMPTS.md (obligatoire selon cahier des charges)
prompts_md = """# PROMPTS.md – Journal des Interactions IA

## Charte IA du Projet
- Rouge (IA interdite) : Tests unitaires, fonctions EDA, première implem NLP
- Orange (structuration) : Configuration DVC/MLflow, débogage
- Vert (libre) : Optimisation, Dockerfile, CI/CD, Monitoring, API

---

## 2025-XX-XX – Module 1 (EDA)
**Prompt :** (écrit manuellement – IA interdite pour EDA de base)
**Justification :** Les fonctions d'exploration ont été écrites sans IA
conformément à la charte rouge.

## 2025-XX-XX – Module 2 (Classification)
**Prompt :** "Explique-moi la différence entre GridSearchCV et Optuna"
**Usage IA :** Structuration seulement (orange) – Choix final fait manuellement
**Critique :** GridSearchCV choisi pour la reproductibilité DVC

## 2025-XX-XX – Module 4 (NLP)
**Prompt :** (écrit manuellement – première implémentation NLP = rouge)
**Justification :** Tokenisation, stopwords, stemming codés sans IA.

## 2025-XX-XX – Module 6 (CI/CD)
**Prompt :** "Génère un workflow GitHub Actions pour lint + tests + Docker"
**Usage IA :** Libre (vert) – optimisation du fichier ci.yml
**Critique :** Le workflow a été vérifié ligne par ligne.

## 2025-XX-XX – Dockerfile
**Prompt :** "Génère un Dockerfile optimisé pour une API FastAPI avec sklearn"
**Usage IA :** Libre (vert)
**Critique :** Image python:3.10-slim choisie pour la légèreté.

---
*Ce fichier doit être complété au fur et à mesure du projet.*
"""

with open('PROMPTS.md', 'w', encoding='utf-8') as f:
    f.write(prompts_md)
print('PROMPTS.md cree')