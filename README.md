# Air Paradis - Analyse de Sentiment avec MLOps

## Description du projet

Ce projet implémente une solution complète d'analyse de sentiments pour la compagnie aérienne fictive "Air Paradis". L'objectif est d'anticiper les potentiels "bad buzz" sur les réseaux sociaux en détectant automatiquement les sentiments (positifs ou négatifs) exprimés dans les tweets concernant la compagnie.

Le projet suit une approche MLOps (Machine Learning Operations) complète, depuis l'expérimentation jusqu'au déploiement, avec suivi des performances et monitoring en production.

## Approches de modélisation

Nous avons exploré trois approches principales pour construire notre système d'analyse de sentiments:

1. **Modèles classiques (Machine Learning traditionnel)**
   - Régression logistique avec TF-IDF
   - Naive Bayes
   - SVM linéaire
   - Random Forest

2. **Modèles sur mesure avancés (Deep Learning)**
   - CNN avec embeddings entraînables
   - CNN avec embeddings GloVe pré-entraînés
   - LSTM avec embeddings entraînables
   - LSTM avec embeddings GloVe pré-entraînés

3. **Modèles transformers (état de l'art)**
   - BERT Base
   - DistilBERT Base (version allégée de BERT)

## Démarche MLOps

### 1. Exploration et préparation des données

- Analyse du jeu de données Sentiment140
- Prétraitement et nettoyage des tweets
- Division en ensembles d'entraînement, validation et test

### 2. Expérimentation et tracking avec MLflow

Nous utilisons MLflow pour suivre toutes les expériences:
- Tracking des hyperparamètres
- Enregistrement des métriques (accuracy, precision, recall, F1-score)
- Stockage des artefacts (modèles, visualisations, configurations)
- Comparaison des performances entre les différentes approches

### 3. Sélection et optimisation des modèles

- Évaluation comparative des modèles selon plusieurs critères:
  - Performance (F1 Score principalement)
  - Temps d'entraînement
  - Taille du modèle
  - Vitesse d'inférence
- Optimisation des hyperparamètres
- Analyse des compromis performance/ressources

### 4. Déploiement en production

- Extraction du meilleur modèle depuis MLflow
- Création d'une API avec Flask/FastAPI
- Déploiement sur Azure Web App
- Pipeline CI/CD avec GitHub Actions

### 5. Monitoring et feedback

- Suivi des performances en production
- Collecte des retours utilisateurs
- Détection de dégradation des performances
- Système d'alerte via Azure Application Insights

## Jeu de données Sentiment140

### Présentation du dataset

Ce projet utilise le jeu de données **Sentiment140**, un corpus standard pour l'analyse de sentiment de tweets. Il contient 1,6 million de tweets annotés automatiquement selon leur sentiment (positif ou négatif).

**Caractéristiques du dataset:**
- **Source**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Taille**: 1,6 million de tweets
- **Format**: CSV
- **Langue**: Anglais
- **Période**: Tweets collectés en 2009
- **Annotations**: Automatiques, basées sur les émoticônes

### Structure des données

Le fichier `training.1600000.processed.noemoticon.csv` contient les colonnes suivantes:
- **sentiment**: 0 (négatif) ou 4 (positif) - recodé en 0/1 dans notre traitement
- **id**: ID numérique du tweet
- **date**: Date de publication du tweet
- **flag**: Indicateur interne (non utilisé)
- **user**: Nom d'utilisateur
- **text**: Contenu textuel du tweet (notre feature principale)

### Prétraitement appliqué

Nous avons appliqué les transformations suivantes au dataset:
- Conversion des sentiments de 0/4 à 0/1 (négatif/positif)
- Nettoyage du texte:
  - Suppression des URLs, mentions (@utilisateur) et hashtags
  - Conversion en minuscules
  - Suppression des caractères spéciaux et de la ponctuation
  - Tokenization (découpage en mots)
  - Suppression des mots vides (stop words)
  - Lemmatisation (optionnelle, configurable via `USE_LEMMATIZATION`)

### Échantillonnage

Pour faciliter les expérimentations rapides, nous avons implémenté un système d'échantillonnage configurable:
- Taille d'échantillon définie par `SAMPLE_SIZE` dans la configuration
- Par défaut, utilisation d'un échantillon de 100 000 tweets
- Stratification préservée pour maintenir la distribution des sentiments
- Option d'utiliser le dataset complet en définissant `SAMPLE_SIZE = None`

### Exemple d'utilisation

Le chemin du dataset est défini dans le fichier de configuration:

```python
# config.py
CONFIG = {
    "DATA_PATH": "data/training.1600000.processed.noemoticon.csv",
    "SAMPLE_SIZE": 100000,  # Utiliser None pour le dataset complet
    "USE_LEMMATIZATION": False,
    # Autres paramètres...
}
```

Dans le notebook, le chargement et prétraitement des données est géré par:

```python
# Chargement des données
df = load_sentiment140_data(CONFIG["DATA_PATH"])

# Prétraitement et division en ensembles d'entraînement/test
X_train, X_test, y_train, y_test = preprocess_data(
    df, 
    sample_size=CONFIG["SAMPLE_SIZE"], 
    lemmatize=CONFIG["USE_LEMMATIZATION"]
)
```

### Statistiques du dataset

Après analyse exploratoire, nous avons constaté:

- **Distribution des sentiments**: Équilibrée (50% positifs, 50% négatifs)
- **Longueur moyenne des tweets**: ~78 caractères
- **Nombre moyen de mots par tweet**: ~14 mots
- **Vocabulaire unique: ~300 000 mots** (avant filtrage)

Cette distribution équilibrée des sentiments est idéale pour l'entraînement de modèles de classification, évitant les biais potentiels liés à des classes déséquilibrées.


## Tracking des expérimentations et stockage des modèles avec MLflow

Ce projet intègre MLflow de manière approfondie pour assurer un suivi rigoureux de toutes les expérimentations et centraliser le stockage des modèles, éléments essentiels d'une approche MLOps robuste.

### Tracking détaillé des expérimentations

Pour chaque modèle entraîné (classique, deep learning ou transformer), nous enregistrons:

#### 1. Paramètres de configuration
- Hyperparamètres spécifiques à chaque modèle
- Paramètres d'entraînement (learning rate, batch size, epochs)
- Configuration du prétraitement (lemmatisation, taille d'échantillon)

#### 2. Métriques de performance
- Classification (accuracy, precision, recall, F1-score)
- Performance (temps d'entraînement, temps d'inférence)
- Ressources (taille du modèle en MB)

#### 3. Suivi de l'entraînement par epoch
- Évolution de la perte (loss) et précision (accuracy)
- Comparaison entre ensembles d'entraînement et validation
- Détection précoce du surapprentissage

### Exemple d'implémentation MLflow

Voici un extrait de notre code illustrant l'implémentation de MLflow pour un modèle Transformer:

```python
# Journaliser les résultats dans MLflow
with mlflow.start_run(run_name=display_name):
    # Ajouter une description détaillée du modèle
    description = f"""
    Modèle de classification: {display_name}
    Type: Transformer ({model_name})
    Architecture: {'DistilBERT' if 'distilbert' in model_name.lower() else 'BERT'}
    Learning rate: {learning_rate}
    Weight decay: {weight_decay}
    Prétraitement: {'Avec lemmatisation' if CONFIG.get('USE_LEMMATIZATION', True) else 'Sans lemmatisation'}
    Taille d'échantillon: {CONFIG.get('SAMPLE_SIZE', 'Dataset complet')}
    Date d'entraînement: {time.strftime('%Y-%m-%d %H:%M:%S')}
    """
    mlflow.set_tag("mlflow.note.content", description)

    # Journaliser les paramètres
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # Journaliser les métriques
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    
    # Journaliser l'historique d'entraînement par epoch
    for i, (loss, val_loss) in enumerate(zip(
        history.history['loss'],
        history.history['val_loss']
    )):
        mlflow.log_metric('loss', loss, step=i)
        mlflow.log_metric('val_loss', val_loss, step=i)
        
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            acc = history.history['accuracy'][i]
            val_acc = history.history['val_accuracy'][i]
            mlflow.log_metric('accuracy', acc, step=i)
            mlflow.log_metric('val_accuracy', val_acc, step=i)

    # Enregistrer le modèle et artefacts associés
    mlflow.log_artifacts(model_dir, "bert_model")
    mlflow.log_artifact(tokenizer_path, "tokenizer")
```

### Visualisation et comparaison des expériences

L'interface web MLflow permet d'explorer et comparer facilement toutes les expériences:

```bash
# Lancer l'interface MLflow UI
mlflow ui
```

![Interface MLflow](visualisations/mlflow_ui_example.png)

Cette interface offre:
- Visualisation des métriques et paramètres
- Comparaison côte-à-côte des différents runs
- Exploration des artefacts (modèles, visualisations)
- Filtrage et tri des expériences

### Sélection des meilleurs modèles

MLflow facilite la sélection du meilleur modèle selon différents critères, comme illustré dans notre script d'extraction de modèle:

```python
# Recherche du meilleur modèle selon le F1 Score
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="",
    order_by=["metrics.f1_score DESC"]
)

best_run = runs[0]
best_model_path = os.path.join(best_run.info.artifact_uri, "model")
```

### Bénéfices de notre approche MLflow

Cette intégration MLflow nous permet de:
- Assurer la reproductibilité des expériences
- Faciliter la comparaison des modèles
- Documenter systématiquement le processus d'expérimentation
- Créer un référentiel centralisé des modèles
- Simplifier le déploiement des meilleurs modèles

## Structure du projet

```
air-paradis-sentiment-mlops/
├── notebooks/                    # Notebooks d'expérimentation
│   └── models_training.ipynb     # Entraînement et évaluation des modèles
│
├── src/                          # Code source Python
│   ├── config.py                 # Configuration globale
│   ├── data_processing.py        # Traitement des données
│   ├── mlflow_utils.py           # Intégration avec MLflow
│   ├── model_definitions.py      # Architecture des modèles
│   ├── model_training.py         # Fonctions d'entraînement
│   └── visualization_utils.py    # Visualisations
│
├── models/                       # Structure pour les modèles entraînés
│   ├── classical/                # Modèles ML classiques
│   ├── deeplearning/             # Modèles CNN et LSTM
│   └── bert/                     # Modèles BERT et DistilBERT
│
├── visualisations/               # Visualisations des résultats
├── results/                      # Métriques et résultats d'évaluation
└── data/                         # mettre training.1600000.processed.noemoticon.csv
```

## Résultats et comparaison des modèles

Notre analyse comparative a révélé que:

1. **Modèles classiques**: Offrent une bonne baseline avec une implémentation simple et rapide
   - Meilleur modèle: Régression logistique (F1-Score: ~0.82)
   - Avantages: Rapide à entraîner, peu coûteux en ressources
   - Limites: Performance plafonnée, ne capture pas bien le contexte

2. **Modèles deep learning**: Excellent compromis performance/ressources
   - Meilleur modèle: CNN avec embeddings entraînables (F1-Score: ~0.86)
   - Avantages: Bonne performance, déploiement viable sur plateformes standard
   - Idéal pour les déploiements avec contraintes (ex: Azure F1)

3. **Modèles transformers**: Performances supérieures mais coûteux
   - Meilleur modèle: BERT Base (F1-Score: ~0.89)
   - DistilBERT: Performance légèrement inférieure (~0.88) mais ~40% plus léger
   - Avantages: Compréhension contextuelle supérieure
   - Limites: Taille importante, temps d'inférence plus long

## Utilisation

### Prérequis
- Python 3.10
- pip ou conda

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/ddrache59100/air-paradis-sentiment-mlops.git
cd air-paradis-sentiment-mlops
```

## Configuration de l'environnement

Ce projet utilise un environnement conda pour gérer les dépendances. Deux options sont disponibles pour reproduire l'environnement:

### Option 1: Utiliser conda (recommandé)

```bash
# Créer un nouvel environnement à partir du fichier environment.yml
conda env create -f environment.yml

# Activer l'environnement
conda activate air-paradis-env
```

### Option 2: Utiliser pip

Si vous préférez ne pas utiliser conda, vous pouvez installer les dépendances via pip:

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Note importante

L'exécution du notebook nécessite certains packages spécifiques pour le traitement des données textuelles et le deep learning. L'utilisation de conda est recommandée car il gère mieux les dépendances complexes, notamment pour TensorFlow et les bibliothèques associées.

## Exécution du notebook d'expérimentation

Pour reproduire nos expérimentations avec MLflow:

```bash
# Activer l'environnement
conda activate air-paradis-env  # ou source venv/bin/activate

# Lancer Jupyter Notebook
jupyter notebook notebooks/models_training.ipynb
```

## Visualisation des expériences MLflow

Pour explorer les expériences enregistrées:

```bash
# Lancer l'interface MLflow UI
mlflow ui
```

Puis accéder à l'interface dans votre navigateur à l'adresse: http://localhost:5000


## API de prédiction

L'API d'analyse de sentiment est disponible dans le dossier `api/`. Elle est basée sur le même code que notre API déployée sur Azure Web App via le dépôt GitHub dédié.

### Fonctionnalités de l'API

- Modèle utilisé: CNN avec embeddings entraînables
- Framework: Flask
- Chargement paresseux du modèle
- Validation des entrées
- Monitoring via Azure Application Insights
- Tests unitaires complets

### Tests de l'API

L'API inclut des tests complets:
```bash
cd api
python test_unit.py      # Tests unitaires
python test_api.py       # Tests d'intégration
Exemple de résultat:
```

Exemple de résultat:

```bash
Test 1: I absolutely love this airline! Best flight ever!
Sentiment: Positif
Probabilités: Positif=0.9928, Negatif=0.0072
Temps de prédiction: 0.5567 secondes

Test 2: This is the worst airline experience I've ever had.
Sentiment: Negatif
Probabilités: Positif=0.0677, Negatif=0.9323
Temps de prédiction: 0.0406 secondes
```

### Déploiement continu

L'API bénéficie d'un pipeline CI/CD via GitHub Actions qui:
1. Exécute les tests unitaires
2. Construit l'application 
3. Déploie sur Azure Web App (tier gratuit F1)

Le code complet du déploiement est disponible dans notre dépôt dédié: [air-paradis-sentiment-api-cnn-embed](https://github.com/ddrache59100/air-paradis-sentiment-api-cnn-embed).

Pour plus de détails sur l'API, consultez le fichier `api/README.md`.
