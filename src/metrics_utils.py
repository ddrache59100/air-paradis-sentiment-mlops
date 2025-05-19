# metrics_utils.py

# Imports nécessaires
import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Union, Any
from visualization_utils import plot_confusion_matrix, plot_roc_curve, plot_learning_curves
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from mlflow_utils import log_model_metrics

# ================================
# Métriques et évaluation
# ================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None) -> Dict:
    """
    Calcule les métriques standards pour l'évaluation de modèles.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        y_pred_proba: Probabilités de prédiction (facultatif)
        
    Returns:
        Dictionnaire contenant les métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Ajouter ROC AUC si les probabilités sont disponibles
    if y_pred_proba is not None:
        try:
            # Si binaire
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                metrics['roc_auc'] = auc(fpr, tpr)
        except Exception as e:
            print(f"Erreur lors du calcul de ROC AUC: {e}")
    
    return metrics


def save_model_results(model_name: str, model, metrics: Dict, 
                       predictions: np.ndarray, y_true: np.ndarray, 
                       predictions_proba: Optional[np.ndarray] = None,
                       model_type: str = "classical", history=None, section: str = None,
                       combined_plots: bool = False):
    """
    Sauvegarde tous les résultats d'un modèle de manière standardisée.
    
    Args:
        model_name: Nom du modèle
        model: Modèle entraîné
        metrics: Dictionnaire des métriques
        predictions: Prédictions sur l'ensemble de test
        y_true: Valeurs réelles
        predictions_proba: Probabilités de prédiction (facultatif)
        model_type: Type de modèle (classical, deeplearning, bert, distilbert)
        history: Historique d'entraînement (pour les modèles deep learning)
        section: Préfixe de section pour les noms de fichiers (ex: "3_2" pour section 3.2)
        combined_plots: Si True, génère également une figure combinée avec matrice de confusion et courbe ROC
    """
    # Créer le dossier de résultats
    results_dir = f"results/{model_type}/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Préfixe pour les visualisations
    prefix = f"{section}_" if section else ""
    
    # Dossier pour les visualisations
    vis_dir = "visualisations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sauvegarder les métriques
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Sauvegarder les prédictions
    np.save(f"{results_dir}/predictions.npy", predictions)
    if predictions_proba is not None:
        np.save(f"{results_dir}/predictions_proba.npy", predictions_proba)
    
    # Sauvegarder l'historique si disponible
    if history is not None:
        with open(f"{results_dir}/history.pkl", "wb") as f:
            pickle.dump(history.history, f)
    
    # Générer et sauvegarder les visualisations standard
    # Sauvegarder à la fois dans le dossier résultats et dans le dossier visualisations
    safe_model_name = model_name.replace(' ', '_').lower()
    
    # Matrice de confusion
    plot_confusion_matrix(y_true, predictions, f"{results_dir}/confusion_matrix.png", model_name, show=False)
    plot_confusion_matrix(y_true, predictions, f"{vis_dir}/{prefix}confusion_matrix_{safe_model_name}.png", model_name, show=False)
    
    # Courbe ROC
    if predictions_proba is not None:
        plot_roc_curve(y_true, predictions_proba, f"{results_dir}/roc_curve.png", model_name, show=False)
        plot_roc_curve(y_true, predictions_proba, f"{vis_dir}/{prefix}roc_curve_{safe_model_name}.png", model_name, show=False)
    
    # Courbes d'apprentissage
    if history is not None:
        plot_learning_curves(history, model_name, f"{results_dir}/learning_curves.png", show=False)
        plot_learning_curves(history, model_name, f"{vis_dir}/{prefix}learning_curves_{safe_model_name}.png", show=True)
    
    # Créer une visualisation combinée si demandé
    if combined_plots and predictions_proba is not None:
        # Créer une figure pour les deux graphiques côte à côte
        plt.figure(figsize=(20, 8))
        
        # 1. Matrice de confusion - à gauche
        plt.subplot(1, 2, 1)
        
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_true, predictions)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Définir les labels
        labels = ["Négatif", "Positif"] if len(np.unique(y_true)) == 2 else [str(i) for i in range(len(np.unique(y_true)))]
        
        # Créer la heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        
        # Ajouter les pourcentages
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j+0.5, i+0.6, f"{cm_percent[i, j]:.1f}%", 
                        ha="center", va="bottom", color="red", fontsize=12, alpha=1.0)
                # plt.text(j+0.5, i+0.6, f"{cm_percent[i, j]:.1f}%", 
                #     ha="center", va="top", color="red", fontsize=12, alpha=1.0,
                #     path_effects=[path_effects.withStroke(linewidth=3, foreground="white")])
            
        plt.title(f"Matrice de confusion - {model_name}", fontsize=14)
        plt.xlabel('Prédiction', fontsize=12)
        plt.ylabel('Réalité', fontsize=12)
        
        # 2. Courbe ROC - à droite
        plt.subplot(1, 2, 2)
        
        # Calculer la courbe ROC
        fpr, tpr, _ = roc_curve(y_true, predictions_proba)
        roc_auc = auc(fpr, tpr)
        
        # Tracer la courbe ROC
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Personnaliser le graphique
        plt.title(f"Courbe ROC - {model_name}", fontsize=14)
        plt.xlabel('Taux de faux positifs', fontsize=12)
        plt.ylabel('Taux de vrais positifs', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        # Sauvegarder la figure combinée
        combined_path = f"{results_dir}/combined_results.png"
        vis_combined_path = f"{vis_dir}/{prefix}combined_results_{safe_model_name}.png"
        
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.savefig(vis_combined_path, dpi=300, bbox_inches='tight')
        #plt.close()
        plt.show()
        
        print(f"Visualisation combinée sauvegardée : {vis_combined_path}")
    
    # # # Enregistrer les résultats dans MLflow
    # # with mlflow.start_run(run_name=model_name):
    # #     log_model_metrics(metrics, model, model_name, model_type, results_dir)
    # # Enregistrer les résultats dans MLflow
    # if mlflow_run_id:
    #     # Réutiliser une run MLflow existante
    #     with mlflow.start_run(run_id=mlflow_run_id, nested=True):  # Utiliser nested=True
    #         log_model_metrics(metrics, model, model_name, model_type, results_dir)
    # else:
    #     # Créer une nouvelle run MLflow
    #     with mlflow.start_run(run_name=model_name):
    #         log_model_metrics(metrics, model, model_name, model_type, results_dir)
    
    # Journaliser les métriques dans MLflow
    # IMPORTANT: Ne pas démarrer une nouvelle run MLflow ici
    # Utiliser directement les fonctions de log si une run est active
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                mlflow.log_metric(name, value)
            except:
                pass  # Ignorer les erreurs si aucune run active
    
    # Enregistrer les artefacts
    try:
        mlflow.log_artifacts(results_dir)
    except:
        pass  # Ignorer les erreurs si aucune run active
    
    print(f"Résultats du modèle '{model_name}' sauvegardés dans {results_dir}")
    print(f"Visualisations sauvegardées dans {vis_dir} avec préfixe '{prefix}'")

def save_results_to_json(results: Dict, filename: str):
    """
    Sauvegarde les résultats dans un fichier JSON.
    
    Args:
        results: Dictionnaire contenant les résultats
        filename: Nom du fichier de sortie
    """
    # Convertir les valeurs numpy en types Python standard
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)


def load_results_from_json(filename: str) -> Dict:
    """
    Charge les résultats depuis un fichier JSON.
    
    Args:
        filename: Nom du fichier à charger
        
    Returns:
        Dictionnaire contenant les résultats
    """
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'r') as f:
        return json.load(f)


def analyze_prediction_errors(model, tokenizer, X_test, y_test, num_samples=500):
    """
    Analyse les erreurs de prédiction du modèle sur un échantillon aléatoire.
    
    Args:
        model: Modèle entraîné (BERT ou DistilBERT)
        tokenizer: Tokenizer utilisé pour le modèle
        X_test: Textes de test
        y_test: Étiquettes de test
        num_samples: Nombre d'échantillons à analyser
        
    Returns:
        error_indices: Indices des erreurs
        error_texts: Textes des erreurs
        error_labels: Étiquettes réelles des erreurs
        error_preds: Prédictions erronées
        error_probas: Probabilités des prédictions erronées
    """
    import numpy as np
    import tensorflow as tf
    
    # Échantillonner aléatoirement
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    texts = X_test.iloc[indices].values
    labels = y_test.iloc[indices].values
    
    # Prédire
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="tf")
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    probas = tf.nn.softmax(logits, axis=1).numpy()
    preds = np.argmax(logits, axis=1)
    
    # Identifier les erreurs
    errors = preds != labels
    error_indices = indices[errors]
    error_texts = texts[errors]
    error_labels = labels[errors]
    error_preds = preds[errors]
    error_probas = probas[errors]
    
    print(f"Nombre total d'erreurs dans l'échantillon: {np.sum(errors)} / {len(texts)} ({np.mean(errors)*100:.2f}%)")
    
    # Analyser les types d'erreurs
    fp = np.sum((labels == 0) & (preds == 1))  # Faux positifs
    fn = np.sum((labels == 1) & (preds == 0))  # Faux négatifs
    
    print(f"Faux positifs (prédits positifs alors que négatifs): {fp} ({fp/np.sum(errors)*100:.2f}% des erreurs)")
    print(f"Faux négatifs (prédits négatifs alors que positifs): {fn} ({fn/np.sum(errors)*100:.2f}% des erreurs)")
    
    # Afficher quelques exemples d'erreurs
    print("\nExemples d'erreurs de prédiction:")
    for i in range(min(5, len(error_texts))):
        sentiment = "positif" if error_labels[i] == 1 else "négatif"
        prediction = "positif" if error_preds[i] == 1 else "négatif"
        confidence = error_probas[i][error_preds[i]]
        print(f"Texte: \"{error_texts[i]}\"")
        print(f"Vrai sentiment: {sentiment}")
        print(f"Prédiction: {prediction} (confiance: {confidence:.2f})")
        print("")
    
    # Créer une visualisation
    plt.figure(figsize=(10, 6))
    plt.bar(['Faux positifs', 'Faux négatifs'], [fp, fn], color=['steelblue', 'lightcoral'])
    plt.title('Types d\'erreurs de prédiction', fontsize=14)
    plt.xlabel('Type d\'erreur')
    plt.ylabel('Nombre d\'erreurs')
    for i, v in enumerate([fp, fn]):
        plt.text(i, v + 1, str(v), ha='center')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('visualisations/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution de la confiance des erreurs
    plt.figure(figsize=(10, 6))
    confidences = [error_probas[i][error_preds[i]] for i in range(len(error_probas))]
    plt.hist(confidences, bins=10, color='steelblue', alpha=0.7)
    plt.title('Distribution de la confiance pour les prédictions erronées', fontsize=14)
    plt.xlabel('Confiance')
    plt.ylabel('Nombre d\'erreurs')
    plt.axvline(0.5, color='red', linestyle='--', label='Seuil de décision')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('visualisations/error_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return error_indices, error_texts, error_labels, error_preds, error_probas