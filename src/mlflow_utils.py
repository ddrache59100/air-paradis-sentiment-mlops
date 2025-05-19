# mlflow_utils.py
 
# Imports nécessaires
import os
import mlflow
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any


# ================================
# Configuration et suivi MLflow
# ================================

def setup_mlflow_experiment(experiment_name: str) -> str:
    """
    Configure un experiment MLflow.
    
    Args:
        experiment_name: Nom de l'experiment
        
    Returns:
        ID de l'experiment
    """
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Vérifier si l'experiment existe déjà
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def register_best_model(model_name, run_id, model_type='sklearn'):
    """
    Enregistre le meilleur modèle dans le MLflow Model Registry.
    
    Args:
        model_name: Nom du modèle à enregistrer
        run_id: ID du run MLflow contenant le modèle
        model_type: Type du modèle (sklearn, keras, tensorflow)
    
    Returns:
        Version du modèle enregistré
    """
    import mlflow.pyfunc
    
    # Construire le chemin du modèle
    if model_type == 'sklearn':
        model_path = f"runs:/{run_id}/model"
    elif model_type in ['keras', 'tensorflow']:
        model_path = f"runs:/{run_id}/model"
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    # Enregistrer le modèle
    result = mlflow.register_model(
        model_path,
        model_name
    )
    

    
    print(f"Modèle {model_name} enregistré avec la version {result.version}")
    
    # Transition vers le stage "Production" pour le meilleur modèle
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
    
    print(f"Modèle {model_name} version {result.version} promu en Production")
    
    return result.version

def compare_mlflow_runs(experiment_name, metric_names, max_runs=10):
    """
    Récupère et compare les runs MLflow pour une expérience donnée.
    
    Args:
        experiment_name: Nom de l'expérience MLflow
        metric_names: Liste des métriques à comparer
        max_runs: Nombre maximum de runs à afficher
    
    Returns:
        DataFrame de comparaison des runs
    """
    # Obtenir l'ID de l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Expérience {experiment_name} non trouvée")
        return None
    
    # Récupérer les runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        print("Aucun run trouvé pour cette expérience")
        return None
    
    # Filtrer les colonnes pertinentes
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    metric_cols = [f"metrics.{name}" for name in metric_names if f"metrics.{name}" in runs.columns]
    
    selected_cols = ['run_id', 'start_time', 'tags.mlflow.runName'] + param_cols + metric_cols
    
    # Sélectionner les colonnes disponibles
    available_cols = [col for col in selected_cols if col in runs.columns]
    
    # Créer un DataFrame de comparaison
    comparison_df = runs[available_cols].sort_values(f"metrics.{metric_names[0]}", ascending=False).head(max_runs)
    
    return comparison_df

# def log_model_metrics(metrics: Dict, model=None, model_name: str = None, 
#                       model_type: str = None, artifacts_path: str = None):
#     """
#     Enregistre les métriques d'un modèle dans MLflow.
    
#     Args:
#         metrics: Dictionnaire de métriques
#         model: Modèle entraîné (facultatif)
#         model_name: Nom du modèle (facultatif)
#         model_type: Type de modèle (sklearn, keras, etc.) (facultatif)
#         artifacts_path: Chemin vers les artefacts à enregistrer (facultatif)
#     """
#     # Enregistrer les métriques
#     for name, value in metrics.items():
#         if isinstance(value, (int, float)):
#             mlflow.log_metric(name, value)
    
#     # Enregistrer les paramètres
#     if model_name:
#         mlflow.log_param("model_name", model_name)
#     if model_type:
#         mlflow.log_param("model_type", model_type)
    
#     # Enregistrer le modèle si disponible
#     if model is not None:
#         if model_type == "sklearn":
#             mlflow.sklearn.log_model(model, "model")
#         elif model_type == "keras":
#             mlflow.keras.log_model(model, "model")
#         elif model_type == "tensorflow":
#             mlflow.tensorflow.log_model(model, "model")
    
#     # Enregistrer des artefacts supplémentaires
#     if artifacts_path and os.path.exists(artifacts_path):
#         mlflow.log_artifacts(artifacts_path)

def log_model_metrics(metrics: Dict, model=None, model_name: str = None, 
                      model_type: str = None, artifacts_path: str = None):
    """
    Enregistre les métriques d'un modèle dans MLflow.
    
    Suppose qu'une run MLflow est déjà active.
    """
    # Enregistrer les métriques
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                mlflow.log_metric(name, value)
            except:
                pass  # Ignorer les erreurs si aucune run active
    
    # Enregistrer les paramètres
    if model_name:
        try:
            mlflow.log_param("model_name", model_name)
        except:
            pass
    if model_type:
        try:
            mlflow.log_param("model_type", model_type)
        except:
            pass
    
    # Enregistrer le modèle si disponible
    if model is not None:
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif model_type == "keras":
                mlflow.keras.log_model(model, "model")
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, "model")
        except:
            pass
    
    # Enregistrer des artefacts supplémentaires
    if artifacts_path and os.path.exists(artifacts_path):
        try:
            mlflow.log_artifacts(artifacts_path)
        except:
            pass