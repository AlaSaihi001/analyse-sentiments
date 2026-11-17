# Analyse de sentiments et détection d’émotions

Ce projet étudie la classification de textes en anglais. Il prédit à la fois une émotion parmi 11 catégories et une polarité parmi 3 catégories. Il compare également des modèles de texte à une approche utilisant un graphe reliant les textes qui partagent des mots-clés.

## Présentation

Le projet comprend le nettoyage des données, l’entraînement et l’évaluation de modèles, l’analyse des résultats, ainsi qu’une petite application interactive réalisée avec Streamlit.

| Tâche | Modèle | F1 macro |
|---|---|---:|
| Émotions (11 classes) | SVM linéaire | 81,4 % |
| Émotions (11 classes) | Régression logistique | 80,3 % |
| Polarité (3 classes) | SVM linéaire | 83,8 % |

Ces résultats sont à interpréter avec prudence : certaines étiquettes du jeu de données semblent avoir été attribuées à partir de mots-clés. Le modèle peut donc apprendre des associations de surface plutôt que comprendre le contexte, la négation ou le sarcasme.

## Préparer l’environnement

Les commandes suivantes sont prévues pour **Git Bash sous Windows**, depuis le dossier du projet :

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Ajouter les données

Le fichier de données n’est pas inclus dans ce dépôt. Téléchargez le jeu [Text Emotion sur Kaggle](https://www.kaggle.com/datasets/prajwalnayakat/text-emotion), puis placez le CSV ici :

```text
data/raw/final_dataset.csv
```

Le fichier doit contenir les colonnes `text` et `emotion`. Des précisions sont disponibles dans [data/README.md](data/README.md).

## Exécuter les scripts

Lancez les commandes depuis la racine du projet, dans cet ordre :

```bash
python code/phase1_pipeline.py
python code/phase2_modeling.py
```

La première étape prépare les données et le graphe, puis produit les fichiers nécessaires dans `data/processed/` et les graphiques de la phase 1. La deuxième entraîne et évalue les modèles et complète les graphiques.

## Démarrer l’application

Après avoir exécuté la phase 1, démarrez l’application avec :

```bash
python -m streamlit run app/app.py
```

Ouvrez ensuite l’adresse locale affichée dans le terminal, généralement `http://localhost:8501`. Au premier démarrage, l’application entraîne et enregistre ses modèles à partir des données préparées.

## Contenu du dépôt

- `code/` : scripts d’analyse et notebooks Jupyter.
- `app/` : application de démonstration Streamlit.
- `data/` : instructions sur les données ; les fichiers bruts ne sont pas fournis.
- `plots/` : graphiques des analyses et des évaluations.
- `report/` : rapports du projet.

## Crédits et licence

Projet universitaire de Data Mining (ING Info 2, 2026–2027), réalisé par Ala Eddine Saihi et Jawaher Fakraoui, sous l’encadrement de Mariem Ghzara.

Les données proviennent du jeu « Text Emotion » de prajwalnayakat sur Kaggle et ne sont pas redistribuées ici. Le code est sous licence MIT ; voir [LICENSE](LICENSE).
