# config.py
import os
from IPython.display import Markdown, display

"""
Configuration globale pour le projet d'analyse de sentiments Air Paradis.
Centralise les paramètres et les chemins.
"""

import os

# Paramètres globaux pour le projet
CONFIG = {
    "RANDOM_SEED": 42,
    "TEST_SIZE": 0.2,
    "VALIDATION_SIZE": 0.1,
    "MAX_FEATURES": 50000,  # Nombre max de mots dans le vocabulaire
    "BATCH_SIZE": 64,
    "EPOCHS": 10,
    "MAX_SEQUENCE_LENGTH": 100,  # Longueur maximale des séquences
    "EMBEDDING_DIM": 100,  # Dimension des embeddings
    "DATA_PATH": "data/training.1600000.processed.noemoticon.csv",
    "MLFLOW_TRACKING_URI": "file:./mlruns",
    "EXPERIMENT_NAME": "Air Paradis - Analyse de Sentiment",
    "SAMPLE_SIZE": None,  # Taille de l'échantillon, None = tout le dataset
    "USE_LEMMATIZATION": False  # Utilisation de la lemmatisation, True = oui, False = non
}

# Fonction pour mettre à jour la taille de l'échantillon
def update_sample_size(size=None):
    """
    Met à jour la taille de l'échantillon dans la configuration.
    
    Args:
        size: Taille de l'échantillon, None pour utiliser tout le dataset
    """
    CONFIG["SAMPLE_SIZE"] = size
    print(f"Taille d'échantillon mise à jour: {size if size is not None else 'Dataset complet'}")
    return CONFIG

# Nouvelle fonction pour mettre à jour le paramètre de lemmatisation
def update_lemmatization(use_lemma=False):
    """
    Met à jour l'utilisation de la lemmatisation dans la configuration.
    
    Args:
        use_lemma: Si True, la lemmatisation sera utilisée lors du prétraitement
    """
    CONFIG["USE_LEMMATIZATION"] = use_lemma
    print(f"Utilisation de la lemmatisation mise à jour: {'Activée' if use_lemma else 'Désactivée'}")
    return CONFIG


# Chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualisations")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Création des répertoires nécessaires
def create_directories():
    """Crée tous les répertoires nécessaires pour le projet."""
    dirs = [
        MODELS_DIR, 
        RESULTS_DIR, 
        VISUALIZATIONS_DIR, 
        MLRUNS_DIR,
        os.path.join(MODELS_DIR, "classical"),
        os.path.join(MODELS_DIR, "deeplearning"),
        os.path.join(MODELS_DIR, "bert"),
        os.path.join(RESULTS_DIR, "classical"),
        os.path.join(RESULTS_DIR, "deeplearning"),
        os.path.join(RESULTS_DIR, "bert")
    ]
    
    created = []
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            created.append(d)
    
    return created

# ================================
# Fonctions d'utilitaires générales
# ================================

def print_md(md):
    """
    print en Markdown
    """
    display(Markdown(md))

def create_directory_structure():
    """
    Crée la structure de répertoires nécessaire pour le projet.
    """
    directories = [
        'data',              # Données
        'models',            # Modèles entraînés
        'results',           # Résultats des modèles
        'visualisations',    # Visualisations
        'mlruns',            # Logs MLflow
        'reports',           # Rapports et présentations
    ]
    
    created = []
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            created.append(directory)
            
    # Créer des sous-répertoires pour chaque type de modèle
    # model_types = ['classical', 'deeplearning', 'bert', 'distilbert']
    model_types = ['classical', 'deeplearning', 'bert']
    for model_type in model_types:
        path = os.path.join('models', model_type)
        if not os.path.exists(path):
            os.makedirs(path)
            created.append(path)
            
        results_path = os.path.join('results', model_type)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            created.append(results_path)
    
    return created