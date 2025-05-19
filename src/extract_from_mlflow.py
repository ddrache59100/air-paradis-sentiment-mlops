# extract_from_mlflow.py (version améliorée)
import os
import sys
import logging
import warnings
import shutil

# Supprimer les avertissements
warnings.filterwarnings('ignore')

# Configurer le niveau de journalisation pour désactiver les messages d'info et d'avertissement
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=debug, 1=info, 2=warning, 3=error

# Supprimer les messages de journalisation de TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Désactiver la journalisation des messages AVX/FMA
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Rediriger stderr pendant l'importation de tensorflow
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    import tensorflow as tf
    # Après l'importation, configurer TensorFlow pour être silencieux
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
except:
    pass
finally:
    # Restaurer stderr
    sys.stderr.close()
    sys.stderr = original_stderr

# Maintenant importer le reste des modules
import mlflow
import pickle
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import glob
import os.path as path

def extract_model_from_mlflow(run_id, model_name, target_dir):
    """
    Extrait un modèle stocké dans MLflow et le sauvegarde dans le répertoire cible.
    
    Args:
        run_id: ID de l'exécution MLflow contenant le modèle
        model_name: Nom du modèle à extraire
        target_dir: Répertoire où sauvegarder le modèle
    """
    print(f"Extraction du modèle '{model_name}' depuis MLflow (Run ID: {run_id})...")
    
    # Créer le répertoire cible s'il n'existe pas
    os.makedirs(target_dir, exist_ok=True)
    
    # Vérifier si le modèle existe déjà dans le répertoire cible
    target_file = os.path.join(target_dir, f"{model_name}.keras")
    if os.path.exists(target_file):
        print(f"Le modèle existe déjà dans le répertoire cible: {target_file}")
        return target_file
    
    # Configurer le client MLflow
    client = MlflowClient()
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        # Tenter d'abord de télécharger les artefacts directement
        print("Tentative de téléchargement direct des artefacts...")
        
        try:
            # Vérifier si le modèle est dans le répertoire 'model'
            model_artifacts_dir = client.download_artifacts(run_id, "model")
            if os.path.exists(model_artifacts_dir):
                print(f"Artefacts téléchargés depuis 'model': {model_artifacts_dir}")
                
                # Vérifier si c'est un modèle Keras
                saved_model_path = os.path.join(model_artifacts_dir, "saved_model.pb")
                if os.path.exists(saved_model_path):
                    print("Format Keras/SavedModel détecté")
                    
                    # Tenter de charger et sauvegarder le modèle
                    try:
                        model = tf.keras.models.load_model(model_artifacts_dir)
                        model.save(target_file)
                        print(f"Modèle sauvegardé dans {target_file}")
                        return target_file
                    except Exception as e:
                        print(f"Erreur lors du chargement/sauvegarde du modèle Keras: {e}")
        except Exception as e:
            print(f"Erreur lors du téléchargement de 'model': {e}")
        
        # Si ça n'a pas fonctionné, essayer de charger le modèle avec mlflow.tensorflow
        print(f"Tentative de chargement via MLflow TensorFlow...")
        model = mlflow.tensorflow.load_model(f"runs:/{run_id}/model")
        
        # Sauvegarder le modèle dans le répertoire cible
        print(f"Sauvegarde du modèle dans {target_file}...")
        model.save(target_file)
        
        print(f"✅ Modèle extrait avec succès et sauvegardé dans {target_file}")
        return target_file
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction du modèle depuis MLflow: {e}")
        return None

def extract_cnn_embeddings_model():
    """
    Extrait spécifiquement le meilleur modèle CNN (embeddings entraînables) depuis MLflow,
    en se basant sur les métriques de performance.
    """
    # Configurer MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    
    # Rechercher le modèle CNN (embeddings entraînables)
    print("Recherche du meilleur modèle 'CNN (embeddings entraînables)' dans MLflow...")
    
    # Récupérer toutes les expériences
    experiments = client.search_experiments()
    
    # Variables pour stocker le meilleur run
    best_run_id = None
    best_metric_value = 0.0
    best_experiment_name = ""
    
    # Métriques à rechercher, par ordre de priorité
    metrics_to_check = ["f1_score", "accuracy", "val_accuracy"]
    
    # Parcourir toutes les expériences pour trouver les runs CNN
    for experiment in experiments:
        print(f"Recherche dans l'expérience: {experiment.name}")
        
        # Rechercher les exécutions dans cette expérience avec le nom "CNN (embeddings entraînables)"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.`mlflow.runName` LIKE '%CNN%embed%'"
        )
        
        # Afficher le nombre de runs trouvés
        print(f"  {len(runs)} run(s) trouvé(s) avec 'CNN (embeddings entraînables)' dans le nom")
        
        if runs:
            for run in runs:
                run_metrics = run.data.metrics
                run_id = run.info.run_id
                
                # Chercher la meilleure métrique disponible
                for metric_name in metrics_to_check:
                    if metric_name in run_metrics:
                        metric_value = run_metrics[metric_name]
                        print(f"  Run {run_id}: {metric_name} = {metric_value}")
                        
                        # Vérifier si c'est le meilleur score jusqu'à présent
                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_run_id = run_id
                            best_experiment_name = experiment.name
                            best_metric_name = metric_name
                            print(f"  → Nouveau meilleur modèle trouvé!")
                        break
    
    if best_run_id:
        print(f"\nMeilleur modèle trouvé:")
        print(f"  Expérience: {best_experiment_name}")
        print(f"  Run ID: {best_run_id}")
        print(f"  {best_metric_name}: {best_metric_value}")
        
        # Extraire le modèle
        model_file = extract_model_from_mlflow(
            run_id=best_run_id,
            model_name="cnn_embeddings_entrainables",
            target_dir="models/deeplearning"
        )
        
        if model_file:
            # Renommer le fichier pour correspondre au format attendu
            target_path = "models/deeplearning/cnn_(embeddings_entraînables).keras"
            if model_file != target_path:
                os.rename(model_file, target_path)
                print(f"Fichier renommé: {model_file} -> {target_path}")
                
            print(f"✅ Meilleur modèle CNN (embeddings entraînables) extrait avec succès")
            
            # Extraire le tokenizer
            extract_tokenizer_from_mlflow(run_id=best_run_id)
            
            return True
    else:
        print("❌ Aucun modèle 'CNN (embeddings entraînables)' trouvé dans MLflow")
    
    return False

def extract_distilbert_model():
    """
    Extrait spécifiquement le modèle DistilBERT depuis MLflow,
    en se basant sur les métriques de performance.
    """
    # Configurer MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    
    # Rechercher le modèle DistilBERT
    print("Recherche du meilleur modèle 'DistilBERT Base' dans MLflow...")
    
    # Récupérer toutes les expériences
    experiments = client.search_experiments()
    
    # Liste pour stocker tous les runs avec le meilleur score
    best_runs = []
    best_metric_value = 0.0
    best_metric_name = ""
    
    # Métriques à rechercher, par ordre de priorité
    metrics_to_check = ["f1_score", "accuracy", "val_accuracy"]
    
    # Parcourir toutes les expériences pour trouver les runs DistilBERT
    for experiment in experiments:
        print(f"Recherche dans l'expérience: {experiment.name}")
        
        # Rechercher les exécutions dans cette expérience avec le nom "DistilBERT"
        # Différentes façons d'écrire DistilBERT dans les noms de run
        for search_pattern in ["DistilBERT", "Distilbert", "distil", "DISTIL"]:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.`mlflow.runName` LIKE '%{search_pattern}%'"
            )
            
            # Afficher le nombre de runs trouvés
            if runs:
                print(f"  {len(runs)} run(s) trouvé(s) avec '{search_pattern}' dans le nom")
                
                for run in runs:
                    run_metrics = run.data.metrics
                    run_id = run.info.run_id
                    run_name = run.data.tags.get("mlflow.runName", "Unknown")
                    
                    print(f"  Évaluation de Run {run_id}: {run_name}")
                    
                    # Chercher la meilleure métrique disponible
                    for metric_name in metrics_to_check:
                        if metric_name in run_metrics:
                            metric_value = run_metrics[metric_name]
                            print(f"    {metric_name} = {metric_value}")
                            
                            # Vérifier si c'est le meilleur score jusqu'à présent
                            if metric_value > best_metric_value:
                                best_metric_value = metric_value
                                best_metric_name = metric_name
                                best_runs = [(run_id, experiment.name, run_name)]
                                print(f"    → Nouveau meilleur modèle trouvé!")
                            elif metric_value == best_metric_value:
                                # Ajouter ce run à la liste des meilleurs runs
                                best_runs.append((run_id, experiment.name, run_name))
                                print(f"    → Autre modèle avec le même score trouvé!")
                            break
    
    if best_runs:
        print(f"\nMeilleurs modèles DistilBERT trouvés ({len(best_runs)}):")
        for i, (run_id, exp_name, run_name) in enumerate(best_runs, 1):
            print(f"  {i}. Run ID: {run_id}, Expérience: {exp_name}, Nom: {run_name}")
            print(f"     {best_metric_name}: {best_metric_value}")
        
        # Créer le répertoire cible
        os.makedirs("models/bert", exist_ok=True)
        
        # Essayer chaque run jusqu'à ce qu'un fonctionne
        for run_id, exp_name, run_name in best_runs:
            print(f"\nEssai d'extraction du modèle depuis Run {run_id}...")
            
            # Lister tous les artefacts et chercher le dossier bert_model
            artifacts = client.list_artifacts(run_id)
            for artifact in artifacts:
                print(f"  Artefact trouvé: {artifact.path}")
                
                if artifact.path == "bert_model" or "bert" in artifact.path.lower():
                    try:
                        print(f"Tentative de téléchargement du dossier '{artifact.path}'...")
                        local_path = client.download_artifacts(run_id, artifact.path)
                        
                        # Vérifier si c'est bien un dossier de modèle BERT
                        if os.path.exists(os.path.join(local_path, "config.json")):
                            # Copier vers le répertoire cible
                            target_path = "models/bert/distilbert_base"
                            if os.path.exists(target_path):
                                shutil.rmtree(target_path)
                            shutil.copytree(local_path, target_path)
                            
                            print(f"✅ Modèle DistilBERT copié avec succès vers {target_path}")
                            
                            # Extraire aussi le tokenizer
                            extract_tokenizer_from_mlflow(run_id=run_id)
                            
                            return True
                        else:
                            print(f"Dossier '{artifact.path}' ne semble pas contenir un modèle BERT valide")
                    except Exception as e:
                        print(f"Erreur lors du téléchargement de '{artifact.path}': {e}")
        
        # Si aucun des runs n'a fonctionné, essayer encore avec une approche différente
        print("\nEssai d'extraction alternative des modèles BERT...")
        for run_id, exp_name, run_name in best_runs:
            try:
                # Parcourir tous les artefacts et chercher des indices
                artifacts_root = client.download_artifacts(run_id, "")
                print(f"Téléchargement réussi du dossier d'artefacts racine: {artifacts_root}")
                
                # Rechercher des modèles BERT dans le dossier téléchargé
                bert_folders = []
                for root, dirs, files in os.walk(artifacts_root):
                    for file in files:
                        if file == "config.json":
                            bert_folders.append(root)
                
                if bert_folders:
                    print(f"Trouvé {len(bert_folders)} dossiers potentiels de modèles BERT:")
                    for folder in bert_folders:
                        print(f"  {folder}")
                    
                    # Prendre le premier dossier trouvé
                    source_folder = bert_folders[0]
                    target_path = "models/bert/distilbert_base"
                    if os.path.exists(target_path):
                        shutil.rmtree(target_path)
                    shutil.copytree(source_folder, target_path)
                    
                    print(f"✅ Modèle DistilBERT copié depuis {source_folder} vers {target_path}")
                    return True
                else:
                    print("Aucun dossier de modèle BERT trouvé dans les artefacts téléchargés")
            except Exception as e:
                print(f"Erreur lors de l'extraction alternative: {e}")
        
        # Si on atteint ce point, on n'a pas pu extraire le modèle
        print("\n❌ Aucun des meilleurs modèles n'a pu être extrait.")
    else:
        print("❌ Aucun modèle DistilBERT trouvé dans MLflow")
    
    # Si on atteint ce point, on n'a pas pu extraire le modèle
    print("ERREUR CRITIQUE: Veuillez vous assurer que vos modèles entraînés sont correctement stockés dans MLflow")
    print("Vérifiez que le notebook 2 a bien été exécuté et que les modèles ont été enregistrés dans MLflow")
    
    return False

def extract_tokenizer_from_mlflow(run_id=None):
    """
    Extrait le tokenizer depuis MLflow.
    
    Args:
        run_id: ID de l'exécution MLflow contenant le tokenizer (optionnel)
    """
    print("Recherche du tokenizer dans MLflow...")
    
    tokenizer_path = "models/deeplearning/tokenizer.pkl"
    if os.path.exists(tokenizer_path):
        print(f"Le tokenizer existe déjà: {tokenizer_path}")
        return True
    
    # Si un run_id est fourni, tenter d'extraire le tokenizer depuis ce run
    if run_id:
        client = MlflowClient()
        mlflow.set_tracking_uri("file:./mlruns")
        
        try:
            # Chercher le tokenizer dans les artefacts du run
            artifacts = client.list_artifacts(run_id)
            tokenizer_artifact = None
            
            for artifact in artifacts:
                if "tokenizer" in artifact.path.lower():
                    tokenizer_artifact = artifact.path
                    break
            
            if tokenizer_artifact:
                print(f"Artefact tokenizer trouvé: {tokenizer_artifact}")
                
                # Télécharger le tokenizer
                local_path = client.download_artifacts(run_id, tokenizer_artifact)
                print(f"Tokenizer téléchargé à {local_path}")
                
                # Vérifier si c'est un fichier ou un répertoire
                if os.path.isdir(local_path):
                    print("Le tokenizer est un répertoire, recherche de fichiers .pkl")
                    
                    # Rechercher les fichiers .pkl dans le répertoire
                    pkl_files = glob.glob(os.path.join(local_path, "*.pkl"))
                    
                    if pkl_files:
                        print(f"Fichier(s) pickle trouvé(s): {pkl_files}")
                        
                        # Prendre le premier fichier .pkl
                        tokenizer_file = pkl_files[0]
                        
                        # S'assurer que le répertoire cible existe
                        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                        
                        # Copier le fichier à l'emplacement cible
                        shutil.copy(tokenizer_file, tokenizer_path)
                        print(f"Fichier tokenizer copié: {tokenizer_file} -> {tokenizer_path}")
                        
                        print(f"✅ Tokenizer extrait avec succès et sauvegardé dans {tokenizer_path}")
                        return True
                    else:
                        print("Aucun fichier .pkl trouvé dans le répertoire du tokenizer")
                        
                        # Essayer de trouver d'autres types de fichiers
                        all_files = glob.glob(os.path.join(local_path, "*"))
                        if all_files:
                            print(f"Autres fichiers trouvés: {all_files}")
                            
                            # Prendre le premier fichier trouvé (s'il existe)
                            if len(all_files) > 0:
                                source_file = all_files[0]
                                
                                # S'assurer que le répertoire cible existe
                                os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                                
                                # Copier le fichier à l'emplacement cible
                                shutil.copy(source_file, tokenizer_path)
                                print(f"Fichier copié: {source_file} -> {tokenizer_path}")
                                
                                print(f"✅ Tokenizer extrait avec succès et sauvegardé dans {tokenizer_path}")
                                return True
                else:
                    # C'est un fichier, le copier directement
                    # S'assurer que le répertoire cible existe
                    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                    
                    # Copier le fichier à l'emplacement cible
                    shutil.copy(local_path, tokenizer_path)
                    
                    print(f"✅ Tokenizer extrait avec succès et sauvegardé dans {tokenizer_path}")
                    return True
            else:
                print("⚠️ Aucun tokenizer trouvé dans les artefacts du run")
                
                # Chercher parmi tous les artefacts
                all_artifacts = []
                
                def list_all_artifacts(run_id, path=""):
                    artifacts = client.list_artifacts(run_id, path)
                    for artifact in artifacts:
                        if artifact.is_dir:
                            list_all_artifacts(run_id, artifact.path)
                        else:
                            all_artifacts.append(artifact.path)
                
                list_all_artifacts(run_id)
                print(f"Tous les artefacts disponibles ({len(all_artifacts)}):")
                for artifact in all_artifacts:
                    print(f"  {artifact}")
                
                # Chercher des fichiers qui pourraient être des tokenizers
                for artifact in all_artifacts:
                    if "tokenizer" in artifact.lower() or ".pkl" in artifact.lower():
                        print(f"Fichier potentiel de tokenizer trouvé: {artifact}")
                        
                        try:
                            # Télécharger l'artefact
                            local_path = client.download_artifacts(run_id, artifact)
                            
                            # Copier à l'emplacement cible
                            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                            shutil.copy(local_path, tokenizer_path)
                            
                            print(f"✅ Tokenizer potentiel extrait et sauvegardé dans {tokenizer_path}")
                            return True
                        except Exception as e:
                            print(f"Erreur lors de l'extraction du tokenizer potentiel {artifact}: {e}")
        
        except Exception as e:
            print(f"Erreur lors de l'extraction du tokenizer: {e}")
    
    print("❌ Impossible d'extraire le tokenizer depuis MLflow")
    return False

if __name__ == "__main__":
    print("=== Extraction des modèles depuis MLflow ===")
    
    # Créer les répertoires nécessaires
    for directory in ["models/deeplearning", "models/bert", "models/checkpoints"]:
        os.makedirs(directory, exist_ok=True)
        print(f"Répertoire {directory} créé ou vérifié.")
    
    # Extraire le meilleur modèle CNN (embeddings entraînables)
    cnn_success = extract_cnn_embeddings_model()
    
    # Si le modèle CNN n'a pas pu être extrait, arrêter l'exécution
    if not cnn_success:
        print("❌ ERREUR CRITIQUE: Impossible d'extraire le modèle CNN")
        print("Veuillez exécuter le notebook 2 pour générer et enregistrer les modèles dans MLflow")
        sys.exit(1)
    
    # Extraire le modèle DistilBERT
    distilbert_success = extract_distilbert_model()
    
    # Extraire ou créer le tokenizer (si pas déjà fait)
    tokenizer_success = extract_tokenizer_from_mlflow()
    
    # Résumé
    print("\n=== Résumé ===")
    print(f"CNN (embeddings entraînables): {'✅ Prêt' if cnn_success else '❌ Non disponible'}")
    print(f"DistilBERT Base: {'✅ Prêt' if distilbert_success else '❌ Non disponible'}")
    print(f"Tokenizer: {'✅ Prêt' if tokenizer_success else '❌ Non disponible'}")
    
    if not (cnn_success and tokenizer_success):
        print("\n❌ ERREUR CRITIQUE: Certains composants n'ont pas pu être extraits")
        print("L'application ne peut pas fonctionner correctement sans tous les modèles")
        print("Veuillez exécuter le notebook 2 pour générer et enregistrer les modèles dans MLflow")
        sys.exit(1)
    else:
        print("\n✅ Extraction des modèles terminée")
        print("Pour tester les modèles, exécutez: python inference.py")