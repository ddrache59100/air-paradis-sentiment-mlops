# visualization_utils.py

# Imports nécessaires
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import CONFIG

# ================================
# Visualisations
# ================================

def plot_confusion_matrix(y_true, y_pred, filename: str, model_name: str = None, show: bool = False):
    """
    Génère et sauvegarde une matrice de confusion stylisée.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        filename: Chemin du fichier de sortie
        model_name: Nom du modèle (facultatif)
    """
    plt.figure(figsize=(10, 8))
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculer les pourcentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Définir les labels
    labels = ["Négatif", "Positif"] if len(np.unique(y_true)) == 2 else [str(i) for i in range(len(np.unique(y_true)))]
    
    # Créer la heatmap
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    # Ajouter les pourcentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.6, f"{cm_percent[i, j]:.1f}%", 
                    ha="center", va="bottom", color="red", fontsize=12, alpha=1.0)
            # # plt.text(j+0.5, i+0.6, f"{cm_percent[i, j]:.1f}%", 
            # #     ha="center", va="top", color="red", fontsize=12, alpha=1.0,
            # #     path_effects=[path_effects.withStroke(linewidth=3, foreground="white")])
            # plt.text(j+0.5, i+0.6, f"{cm_percent[i, j]:.1f}%", 
            #     ha="center", va="top", color="red", fontsize=10, alpha=1.0,
            #     path_effects=[path_effects.withStroke(linewidth=3, foreground="white")])
            
    # Ajouter les titres et les étiquettes
    title = f"Matrice de confusion - {model_name}" if model_name else "Matrice de confusion"
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(y_true, y_pred_proba, filename: str, model_name: str = None, show: bool = False):
    """
    Génère et sauvegarde une courbe ROC.
    
    Args:
        y_true: Valeurs réelles
        y_pred_proba: Probabilités de prédiction
        filename: Chemin du fichier de sortie
        model_name: Nom du modèle (facultatif)
    """
    plt.figure(figsize=(10, 8))
    
    # Calculer la courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Tracer la courbe ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Ajouter les titres et les étiquettes
    title = f"Courbe ROC - {model_name}" if model_name else "Courbe ROC"
    plt.title(title, fontsize=14)
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Taux de vrais positifs', fontsize=12)
    
    # Personnaliser le graphique
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(history, model_name: str, filename: str, show: bool = False):
    """
    Génère et sauvegarde les courbes d'apprentissage avec mise en évidence du surapprentissage.
    
    Args:
        history: Historique d'entraînement Keras
        model_name: Nom du modèle
        filename: Chemin du fichier de sortie
    """
    plt.figure(figsize=(15, 6))
    
    # Vérifier si l'historique contient des données de validation
    has_val = 'val_accuracy' in history.history and 'val_loss' in history.history
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'o-', label='Entraînement')
    
    if has_val:
        plt.plot(history.history['val_accuracy'], 'o-', label='Validation')
        
        # Calcul de l'écart de surapprentissage
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        acc_gap = train_acc - val_acc
        
        plt.title(f'Accuracy - Écart: {acc_gap:.4f}')
        plt.axhline(y=val_acc, color='r', linestyle='--', alpha=0.3)
        plt.fill_between(range(len(history.history['accuracy'])), 
                          history.history['accuracy'], 
                          history.history['val_accuracy'],
                          alpha=0.1, color='red')
    else:
        plt.title('Accuracy')
    
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'o-', label='Entraînement')
    
    if has_val:
        plt.plot(history.history['val_loss'], 'o-', label='Validation')
        
        # Calcul de l'écart de surapprentissage
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        loss_ratio = val_loss / train_loss
        
        plt.title(f'Loss - Ratio Val/Train: {loss_ratio:.2f}')
        plt.axhline(y=val_loss, color='r', linestyle='--', alpha=0.3)
        
        # Mettre en évidence la zone de surapprentissage
        plt.fill_between(range(len(history.history['loss'])), 
                         history.history['loss'], 
                         history.history['val_loss'],
                         where=(np.array(history.history['val_loss']) > np.array(history.history['loss'])),
                         alpha=0.1, color='red')
    else:
        plt.title('Loss')
    
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(f"Courbes d'apprentissage - {model_name}", fontsize=16)
    plt.tight_layout()
    
    # Sauvegarder la figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(results_df, metrics=None, filename=None):
    """
    Génère un graphique comparatif des performances de différents modèles.
    
    Args:
        results_df: DataFrame contenant les résultats des modèles
        metrics: Liste des métriques à afficher (facultatif)
        filename: Chemin du fichier de sortie (facultatif)
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Augmenter légèrement la hauteur pour avoir de l'espace pour la légende entre le titre et le graphique
    plt.figure(figsize=(14, 7))
    
    # Préparation des données pour le graphique
    df_melted = pd.melt(results_df, id_vars=['Modèle', 'Type'], 
                        value_vars=metrics, 
                        var_name='Métrique', value_name='Score')
    
    # Création du graphique à barres mais avec plus d'espace en haut
    ax = plt.subplot(111)
    sns.barplot(x='Modèle', y='Score', hue='Métrique', data=df_melted, ax=ax)
    
    # Personnalisation du graphique
    plt.title('Comparaison des performances des modèles', fontsize=16, pad=50)  # Augmenter le padding du titre
    plt.xlabel('Modèle', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha='right')
    
    # Placer la légende sous le titre et au-dessus du graphique
    # Utiliser 'upper center' avec une valeur y positive mais moins que 1
    legend = plt.legend(title='Métrique', loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5)
    
    # Déplacer la légende pour qu'elle soit juste sous le titre
    plt.gca().add_artist(legend)
    
    # Ajout d'une grille pour plus de lisibilité
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuster pour laisser de l'espace en haut
    
    # Sauvegarder ou afficher la figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_time_comparison(results_df, filename=None):
    """
    Génère un graphique comparatif des temps d'entraînement des modèles.
    
    Args:
        results_df: DataFrame contenant les résultats des modèles
        filename: Chemin du fichier de sortie (facultatif)
    """
    plt.figure(figsize=(12, 6))
    
    # Création du graphique à barres
    # sns.barplot(x='Modèle', y='Temps (s)',               data=results_df, palette='viridis')
    sns.barplot(x='Modèle', y='Temps (s)', hue='Modèle', data=results_df, palette='crest', legend=False)
    # Personnalisation du graphique
    plt.title("Temps d'entraînement des modèles (Échelle logarithmique)", fontsize=16)
    plt.xlabel('Modèle', fontsize=14)
    plt.ylabel('Temps (secondes)', fontsize=14)
    plt.yscale('log')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(results_df['Temps (s)']):
        if v < 60:
            label = f"{v:.2f}s"
        elif v < 3600:
            label = f"{v/60:.2f}min"
        else:
            label = f"{v/3600:.2f}h"
        plt.text(i, v*1.002, label, ha='center', va='bottom', fontsize=10, rotation=0)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder ou afficher la figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_size_comparison(results_df, filename=None):
    """
    Génère un graphique comparatif des tailles des modèles.
    
    Args:
        results_df: DataFrame contenant les résultats des modèles
        filename: Chemin du fichier de sortie (facultatif)
    """
    plt.figure(figsize=(12, 6))
    
    # Création du graphique à barres
    # sns.barplot(x='Modèle', y='Taille (MB)', data=results_df, palette='crest')
    sns.barplot(x='Modèle', y='Taille (MB)', hue='Modèle', data=results_df, palette='crest', legend=False)
    # Personnalisation du graphique
    plt.title("Taille des modèles (Échelle logarithmique)", fontsize=16)
    plt.xlabel('Modèle', fontsize=14)
    plt.ylabel('Taille (MB)', fontsize=14)
    plt.yscale('log')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(results_df['Taille (MB)']):
        plt.text(i, v*1.002, f"{v:.1f} MB", ha='center', va='bottom', fontsize=10, rotation=0)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder ou afficher la figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_radar_comparison(results_df, filename=None):
    """
    Génère un graphique radar pour comparer les modèles sur plusieurs dimensions.
    """
    # Préparer les données
    radar_df = results_df[['Modèle', 'F1 Score', 'Temps (s)', 'Taille (MB)']].copy()
    radar_df['Vitesse'] = 1 - (np.log10(radar_df['Temps (s)'] + 1) / np.log10(radar_df['Temps (s)'].max() + 1))
    radar_df['Compacité'] = 1 - (np.log10(radar_df['Taille (MB)'] + 1) / np.log10(radar_df['Taille (MB)'].max() + 1))
    
    categories = ['Performance (F1)', 'Vitesse d\'entraînement', 'Compacité']
    n_cats = len(categories)
    
    # Création d'une figure
    plt.figure(figsize=(10, 8))
    
    # Position des axes
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone
    
    # Ajouter les axes
    ax = plt.subplot(111, polar=True)
    
    # Ajouter les labels d'axes
    plt.xticks(angles[:-1], categories, size=12)
    
    # Retirer les labels des rayons
    ax.set_yticklabels([])
    
    # Assurer que le graphique est centré et visible
    ax.set_ylim(0, 1.2)  # Augmenter la limite pour s'assurer que tout est visible
    
    # Tracer les polygones pour chaque modèle
    for i, row in radar_df.iterrows():
        values = [row['F1 Score'], row['Vitesse'], row['Compacité']]
        values += values[:1]  # Fermer le polygone
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Modèle'])
        ax.fill(angles, values, alpha=0.1)
    
    # Ajouter une légende en bas à droite, à l'extérieur du graphique
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    # Titre du graphique
    plt.title('Comparaison multidimensionnelle des modèles', size=15)
    
    # Ajuster la mise en page pour s'assurer que tout est visible
    plt.tight_layout()
    
    # Sauvegarder ou afficher la figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_analysis(model, tokenizer, X_test, y_test, model_type='transformer', max_examples=10):
    """
    Analyse et visualise les erreurs de prédiction du modèle.
    
    Args:
        model: Modèle entraîné (BERT ou DistilBERT)
        tokenizer: Tokenizer utilisé pour le modèle
        X_test: Textes de test
        y_test: Étiquettes de test
        model_type: Type de modèle ('transformer')
        max_examples: Nombre maximal d'exemples à afficher
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    
    # Sélectionner un échantillon aléatoire
    random_indices = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
    sample_texts = X_test.iloc[random_indices].values
    sample_labels = y_test.iloc[random_indices].values
    
    # Prédire avec le modèle
    inputs = tokenizer(list(sample_texts), return_tensors="tf", padding=True, truncation=True)
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    predictions = np.argmax(logits, axis=1)
    
    # Trouver les erreurs
    errors = predictions != sample_labels
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("Aucune erreur trouvée dans l'échantillon.")
        return
    
    # Limiter le nombre d'exemples
    error_indices = error_indices[:max_examples]
    
    # Créer un DataFrame pour l'analyse
    error_data = []
    for idx in error_indices:
        text = sample_texts[idx]
        true_label = "Positif" if sample_labels[idx] == 1 else "Négatif"
        pred_label = "Positif" if predictions[idx] == 1 else "Négatif"
        proba = tf.nn.softmax(logits, axis=1).numpy()[idx]
        confidence = proba[predictions[idx]]
        
        error_data.append({
            "Texte": text,
            "Vrai sentiment": true_label,
            "Prédiction": pred_label,
            "Confiance": confidence,
            "Texte tronqué": text[:50] + "..." if len(text) > 50 else text
        })
    
    error_df = pd.DataFrame(error_data)
    
    # Afficher le tableau des erreurs
    print(f"\nAnalyse des erreurs de prédiction ({len(error_indices)} exemples):")
    pd.set_option('display.max_colwidth', 50)
    display(error_df[["Texte tronqué", "Vrai sentiment", "Prédiction", "Confiance"]])
    pd.reset_option('display.max_colwidth')
    
    # Visualiser la distribution des erreurs
    plt.figure(figsize=(10, 6))
    confusion = pd.crosstab(
        pd.Series([f"Vrai: {x}" for x in error_df["Vrai sentiment"]], name="Vrai"),
        pd.Series([f"Prédit: {x}" for x in error_df["Prédiction"]], name="Prédit")
    )
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Types d'erreurs de prédiction", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"visualisations/error_analysis_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualiser la distribution de la confiance des erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(error_df["Confiance"], bins=10, kde=True)
    plt.axvline(0.5, color='red', linestyle='--', label="Seuil de décision")
    plt.title("Distribution de la confiance pour les prédictions erronées", fontsize=14)
    plt.xlabel("Confiance")
    plt.ylabel("Nombre d'erreurs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"visualisations/error_confidence_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_visualization(model, tokenizer, text, layer_idx=-1, head_idx=0):
    """
    Visualise l'attention du modèle Transformer sur un texte donné.
    
    Args:
        model: Modèle BERT ou DistilBERT
        tokenizer: Tokenizer associé au modèle
        text: Texte à analyser
        layer_idx: Index de la couche d'attention (-1 pour la dernière)
        head_idx: Index de la tête d'attention (0 par défaut)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import tensorflow as tf
    
    # Tokeniser le texte
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer(text, return_tensors="tf")
    
    # Extraction des sorties d'attention
    if hasattr(model, 'bert'):
        # Pour BERT
        outputs = model(inputs, output_attentions=True)
        attention = outputs.attentions
        model_type = "BERT"
    elif hasattr(model, 'distilbert'):
        # Pour DistilBERT
        outputs = model(inputs, output_attentions=True)
        attention = outputs.attentions
        model_type = "DistilBERT"
    else:
        print("Modèle non compatible avec l'extraction d'attention")
        return
    
    # Récupérer les poids d'attention
    # attention est un tuple de tenseurs, chaque tenseur correspond à une couche
    if attention is not None and len(attention) > 0:
        # Sélectionner la couche
        layer = attention[layer_idx]
        # Convertir en numpy pour le traitement
        attention_data = layer.numpy()
        # Sélectionner la tête d'attention
        head_data = attention_data[0, head_idx, :, :]
        
        # Créer la matrice d'attention (filtrer les tokens spéciaux si nécessaire)
        plt.figure(figsize=(10, 8))
        sns.heatmap(head_data, cmap="viridis", xticklabels=tokens, yticklabels=tokens)
        plt.title(f"Matrice d'attention - {model_type} - Couche {layer_idx+1} - Tête {head_idx+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"visualisations/attention_{model_type.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Aucune donnée d'attention disponible")


def compare_all_models(classical_results, dl_results, bert_results, use_lemmatization=True):
    """
    Compare tous les modèles entraînés.
    
    Args:
        classical_results: Résultats des modèles classiques
        dl_results: Résultats des modèles deep learning
        bert_results: Résultats des modèles BERT/DistilBERT
        use_lemmatization: Indique si la lemmatisation a été utilisée
        
    Returns:
        all_results: DataFrame avec tous les résultats
        all_results_sorted: DataFrame avec tous les résultats triés par score global
    """
    import numpy as np
    import pandas as pd
    
    # Réinitialiser les indices avant de combiner
    classical_results_reset = classical_results.reset_index(drop=True)
    dl_results_reset = dl_results.reset_index(drop=True)
    bert_results_reset = bert_results.reset_index(drop=True)
    
    # Combiner tous les résultats
    all_results = pd.concat([
        classical_results_reset, 
        dl_results_reset, 
        bert_results_reset
    ], ignore_index=True)
    
    # Ajouter l'information de lemmatisation à tous les modèles
    all_results['Lemmatisation'] = 'Oui' if use_lemmatization else 'Non'
    
    # Calculer un score de compromis
    if not all_results.empty:
        # Calculer un score de compromis
        all_results['Performance'] = all_results['F1 Score']
        all_results['Speed'] = 1 / np.log10(all_results['Temps (s)'] + 1)  # Inverse du log pour avoir un score plus élevé pour les modèles rapides
        all_results['Compactness'] = 1 / np.log10(all_results['Taille (MB)'] + 1)  # Inverse du log pour avoir un score plus élevé pour les modèles compacts

        # Normaliser les scores entre 0 et 1
        for col in ['Performance', 'Speed', 'Compactness']:
            min_val = all_results[col].min()
            max_val = all_results[col].max()
            if max_val > min_val:  # Éviter la division par zéro
                all_results[col] = (all_results[col] - min_val) / (max_val - min_val)
            else:
                all_results[col] = 0.5  # Valeur par défaut si tous les modèles ont la même valeur

        # Calculer un score global (en donnant plus de poids à la performance)
        all_results['Global_Score'] = (0.6 * all_results['Performance'] + 
                                      0.2 * all_results['Speed'] + 
                                      0.2 * all_results['Compactness'])

        # Trier par score global
        all_results_sorted = all_results.sort_values('Global_Score', ascending=False)
    else:
        all_results_sorted = all_results
    
    return all_results, all_results_sorted



def create_comparison_visualizations(all_results, all_results_sorted=None):
    """
    Crée différentes visualisations pour comparer tous les modèles.
    
    Args:
        all_results: DataFrame contenant tous les résultats
        all_results_sorted: DataFrame trié par score global (optionnel)
    """
    if all_results_sorted is None:
        all_results_sorted = all_results
    
    try:
        # Visualiser les performances des modèles
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Modèle', y='F1 Score', hue='Type', data=all_results)
        plt.title('Comparaison des F1 Scores', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualisations/all_models_performance_comparison.png')
        plt.show()
        plt.close()
        
        # Visualiser le rapport performance/temps
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=all_results,
            x='Temps (s)',
            y='F1 Score',
            size='Taille (MB)',
            hue='Type',
            sizes=(50, 500),
            alpha=0.7
        )
        plt.xscale('log')
        plt.title('Rapport Performance/Temps/Taille', fontsize=16)
        plt.xlabel('Temps d\'entraînement (secondes, échelle log)', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Annoter chaque point avec le nom du modèle
        for i, row in all_results.iterrows():
            plt.annotate(
                row['Modèle'],
                (row['Temps (s)'], row['F1 Score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.tight_layout()
        plt.savefig('visualisations/performance_time_size_relation.png')
        plt.show()
        plt.close()
        
        # Visualiser le score global si disponible
        if 'Global_Score' in all_results_sorted.columns:
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=all_results_sorted,
                x='Modèle',
                y='Global_Score',
                hue='Type',
                palette='viridis'
            )
            plt.title('Score Global (60% Performance, 20% Vitesse, 20% Compacité)', fontsize=16)
            plt.xlabel('Modèle', fontsize=14)
            plt.ylabel('Score Global', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('visualisations/global_score_comparison.png')
            plt.show()
            plt.close()
        
    except Exception as e:
        print(f"Erreur lors de la création des visualisations: {e}")
        
        # Visualisation simplifiée en cas d'erreur
        plt.figure(figsize=(10, 6))
        plt.bar(all_results['Modèle'], all_results['F1 Score'])
        plt.title('Comparaison des F1 Scores')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualisations/all_models_simple_comparison.png')
        plt.show()
        plt.close()


def generate_model_recommendation(all_results):
    """
    Génère une recommandation détaillée basée sur les résultats des modèles.
    
    Args:
        all_results: DataFrame contenant les résultats de tous les modèles
        
    Returns:
        Texte de recommandation en format Markdown
    """
    # Vérifier si le DataFrame n'est pas vide
    if all_results.empty:
        return "Aucun résultat de modèle disponible pour générer une recommandation."
    
    # Trouver le meilleur modèle global selon le score global
    best_model = all_results.iloc[0]
    best_model_name = best_model['Modèle']
    best_model_type = best_model['Type']
    
    # Trouver le meilleur modèle de chaque type
    best_classical = all_results[all_results['Type'] == 'Classique'].iloc[0] if any(all_results['Type'] == 'Classique') else None
    best_dl = all_results[all_results['Type'] == 'Deep Learning'].iloc[0] if any(all_results['Type'] == 'Deep Learning') else None
    best_transformer = all_results[all_results['Type'] == 'Transformer'].iloc[0] if any(all_results['Type'] == 'Transformer') else None
    
    # Générer la recommandation
    recommendation = f"""
### Recommandation pour Air Paradis

#### Modèle recommandé: **{best_model_name}** ({best_model_type})

**Caractéristiques:**
- F1 Score: **{best_model['F1 Score']:.4f}**
- Temps d'entraînement: **{best_model['Temps (s)']:.2f}** secondes
- Taille du modèle: **{best_model['Taille (MB)']:.2f}** MB
- Score global: **{best_model['Global_Score']:.4f}**

**Préparation des données:**
- Lemmatisation: **{"Activée" if CONFIG.get("USE_LEMMATIZATION", True) else "Désactivée"}**
- Taille de l'échantillon: **{CONFIG.get("SAMPLE_SIZE", "Dataset complet")}**

**Comparaison avec les autres approches:**
"""
    
    # Ajouter les informations sur les autres types de modèles
    if best_classical is not None and best_model_type != 'Classique':
        recommendation += f"""
- **Meilleur modèle classique** ({best_classical['Modèle']}):
  - F1 Score: {best_classical['F1 Score']:.4f} ({(best_model['F1 Score'] - best_classical['F1 Score']) / best_classical['F1 Score'] * 100:.1f}% supérieur)
  - {best_classical['Temps (s)']:.2f}s d'entraînement ({(best_classical['Temps (s)'] / best_model['Temps (s)']):.1f}x plus rapide)
  - {best_classical['Taille (MB)']:.2f} MB ({(best_classical['Taille (MB)'] / best_model['Taille (MB)']):.2f}x plus compact)
"""
    
    if best_dl is not None and best_model_type != 'Deep Learning':
        recommendation += f"""
- **Meilleur modèle deep learning** ({best_dl['Modèle']}):
  - F1 Score: {best_dl['F1 Score']:.4f} ({(best_model['F1 Score'] - best_dl['F1 Score']) / best_dl['F1 Score'] * 100:.1f}% supérieur)
  - {best_dl['Temps (s)']:.2f}s d'entraînement ({(best_dl['Temps (s)'] / best_model['Temps (s)']):.1f}x plus rapide)
  - {best_dl['Taille (MB)']:.2f} MB ({(best_dl['Taille (MB)'] / best_model['Taille (MB)']):.2f}x plus compact)
"""
    
    if best_transformer is not None and best_model_type != 'Transformer':
        recommendation += f"""
- **Meilleur modèle Transformer** ({best_transformer['Modèle']}):
  - F1 Score: {best_transformer['F1 Score']:.4f} ({(best_model['F1 Score'] - best_transformer['F1 Score']) / best_transformer['F1 Score'] * 100:.1f}% supérieur)
  - {best_transformer['Temps (s)']:.2f}s d'entraînement ({(best_transformer['Temps (s)'] / best_model['Temps (s)']):.1f}x plus rapide)
  - {best_transformer['Taille (MB)']:.2f} MB ({(best_transformer['Taille (MB)'] / best_model['Taille (MB)']):.2f}x plus compact)
"""
    
    # Ajouter la justification de la recommandation
    recommendation += """
#### Justification:

"""
    
    # Déterminer la justification en fonction du type de modèle
    if best_model_type == 'Classique':
        recommendation += f"""
Le modèle **{best_model_name}** offre le meilleur compromis entre performance et efficacité. Bien que les modèles deep learning et Transformer obtiennent des scores légèrement supérieurs, ce modèle classique est beaucoup plus rapide à entraîner, moins gourmand en ressources et plus facile à déployer en production.

Pour Air Paradis, qui souhaite mettre rapidement en place un système de détection de bad buzz, ce modèle représente un excellent point de départ avec une très bonne performance (F1 Score de {best_model['F1 Score']:.4f}).
"""
    elif best_model_type == 'Deep Learning':
        recommendation += f"""
Le modèle **{best_model_name}** représente un excellent équilibre entre performance et complexité. Il offre une amélioration significative par rapport aux modèles classiques, tout en restant plus léger et plus rapide que les modèles Transformer.

Pour Air Paradis, ce modèle offre une solution robuste avec une excellente précision (F1 Score de {best_model['F1 Score']:.4f}) tout en maintenant des temps d'inférence rapides, ce qui est crucial pour surveiller les réseaux sociaux en temps réel.
"""
    else:  # Transformer
        recommendation += f"""
Le modèle **{best_model_name}** offre les meilleures performances de tous les modèles testés. Grâce à sa capacité à comprendre le contexte et les nuances du langage, il est particulièrement adapté pour détecter les sentiments subtils dans les tweets.

Pour Air Paradis, l'investissement dans ce modèle plus complexe est justifié par sa capacité supérieure à identifier précisément les bad buzz potentiels (F1 Score de {best_model['F1 Score']:.4f}), ce qui permettra à l'entreprise de réagir plus efficacement aux commentaires négatifs sur les réseaux sociaux.
"""
    
    # Ajouter des conseils pour le déploiement
    recommendation += """
#### Stratégie de déploiement recommandée:

1. **Phase initiale:** Déployer le modèle recommandé en production avec surveillance continue des performances.
2. **Collecte de feedback:** Recueillir les prédictions incorrectes signalées par les utilisateurs pour enrichir le dataset.
3. **Amélioration continue:** Réentraîner périodiquement le modèle avec de nouvelles données spécifiques à Air Paradis.
4. **Test A/B:** Comparer les performances du modèle déployé avec des versions améliorées avant de les promouvoir en production.
"""
    
    # Mentionner l'impact de la lemmatisation
    recommendation += f"""
#### Impact de la lemmatisation:

La préparation des données avec la lemmatisation **{"activée" if CONFIG.get("USE_LEMMATIZATION", True) else "désactivée"}** a contribué aux performances du modèle en {"réduisant la dimensionnalité du vocabulaire et en améliorant la généralisation" if CONFIG.get("USE_LEMMATIZATION", True) else "préservant les formes exactes des mots, ce qui peut être important pour certains types d'expressions sur les réseaux sociaux"}.

Pour optimiser davantage les performances, des tests comparatifs avec {"et" if CONFIG.get("USE_LEMMATIZATION", True) else "ou"} sans lemmatisation pourraient être envisagés lors des prochaines itérations.
"""
    
    return recommendation


# Fonction pour charger des résultats avec gestion d'erreurs
def load_results_safely(path, default_model_type=None):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # S'assurer que les colonnes nécessaires existent
            required_cols = ['Modèle', 'Type', 'F1 Score', 'Temps (s)', 'Taille (MB)']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Colonnes manquantes dans {path}: {missing_cols}")
                return pd.DataFrame(columns=required_cols)
            
            return df
        else:
            print(f"Fichier {path} non trouvé")
            return pd.DataFrame(columns=['Modèle', 'Type', 'F1 Score', 'Temps (s)', 'Taille (MB)'])
    except Exception as e:
        print(f"Erreur lors du chargement de {path}: {e}")
        return pd.DataFrame(columns=['Modèle', 'Type', 'F1 Score', 'Temps (s)', 'Taille (MB)'])


def calculate_global_score(all_results, perf_weight=0.6, speed_weight=0.2, compact_weight=0.2):
    """
    Calcule un score global pour chaque modèle en combinant performance, vitesse et compacité.
    
    Args:
        all_results: DataFrame contenant les résultats de tous les modèles
        perf_weight: Poids accordé à la performance (F1 Score) dans le score global (défaut: 0.6)
        speed_weight: Poids accordé à la vitesse d'entraînement dans le score global (défaut: 0.2)
        compact_weight: Poids accordé à la compacité du modèle dans le score global (défaut: 0.2)
        
    Returns:
        all_results_scored: DataFrame contenant les résultats avec le score global calculé
        all_results_sorted: DataFrame trié par score global décroissant
    """
    # Vérification que les poids somment à 1
    total_weight = perf_weight + speed_weight + compact_weight
    if abs(total_weight - 1.0) > 1e-10:
        print(f"Attention: La somme des poids ({total_weight}) n'est pas égale à 1. Les poids seront normalisés.")
        perf_weight /= total_weight
        speed_weight /= total_weight
        compact_weight /= total_weight
    
    # Création d'une copie pour ne pas modifier l'original
    all_results_scored = all_results.copy()
    
    # Calcul des métriques transformées
    all_results_scored['Performance'] = all_results_scored['F1 Score']
    all_results_scored['Speed'] = 1 / np.log10(all_results_scored['Temps (s)'] + 1)  # Inverse du log pour les modèles rapides
    all_results_scored['Compactness'] = 1 / np.log10(all_results_scored['Taille (MB)'] + 1)  # Inverse du log pour les modèles compacts

    # Normalisation des scores entre 0 et 1
    for col in ['Performance', 'Speed', 'Compactness']:
        min_val = all_results_scored[col].min()
        max_val = all_results_scored[col].max()
        if max_val > min_val:  # Éviter la division par zéro
            all_results_scored[col] = (all_results_scored[col] - min_val) / (max_val - min_val)
        else:
            all_results_scored[col] = 0.5  # Valeur par défaut si tous les modèles ont la même valeur

    # Calcul du score global pondéré
    all_results_scored['Global_Score'] = (
        perf_weight * all_results_scored['Performance'] + 
        speed_weight * all_results_scored['Speed'] + 
        compact_weight * all_results_scored['Compactness']
    )
    
    # Tri par score global décroissant
    all_results_sorted = all_results_scored.sort_values('Global_Score', ascending=False)
    
    # Affichage de la formule utilisée
    print(f"\nFormule du score global: {perf_weight:.1f} × Performance + {speed_weight:.1f} × Vitesse + {compact_weight:.1f} × Compacité")
    
    return all_results_scored, all_results_sorted


def visualize_global_score(all_results_sorted):
    """
    Crée une visualisation du score global et ses composantes pour chaque modèle.
    
    Args:
        all_results_sorted: DataFrame trié par score global contenant les modèles à visualiser
    """
    # Préparation des données pour la visualisation
    models = all_results_sorted['Modèle']
    types = all_results_sorted['Type']
    
    # Créer un DataFrame pour la visualisation
    viz_data = pd.DataFrame({
        'Modèle': np.repeat(models, 3),
        'Type': np.repeat(types, 3),
        'Composante': np.tile(['Performance', 'Vitesse', 'Compacité'], len(models)),
        'Score': np.concatenate([
            all_results_sorted['Performance'].values,
            all_results_sorted['Speed'].values,
            all_results_sorted['Compactness'].values
        ])
    })
    
    # Visualiser le score global
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=all_results_sorted,
        x='Modèle',
        y='Global_Score',
        hue='Type',
        palette='viridis'
    )
    plt.title('Score Global (Performance, Vitesse, Compacité)', fontsize=16)
    plt.xlabel('Modèle', fontsize=14)
    plt.ylabel('Score Global', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualisations/global_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualiser les composantes du score
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=viz_data,
        x='Modèle',
        y='Score',
        hue='Composante',
        palette='Set2'
    )
    plt.title('Composantes du Score Global par Modèle', fontsize=16)
    plt.xlabel('Modèle', fontsize=14)
    plt.ylabel('Score Normalisé', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title='Composante')
    plt.tight_layout()
    plt.savefig('visualisations/global_score_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Graphique radar pour visualiser les 3 dimensions
    # Sélectionner un sous-ensemble de modèles pour la lisibilité (les 5 meilleurs)
    top_models = all_results_sorted.head(5)
    
    # Préparer les données pour le radar
    categories = ['Performance', 'Vitesse', 'Compacité']
    N = len(categories)
    
    # Créer l'angle pour chaque catégorie
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Initialiser la figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Pour chaque modèle
    for i, row in top_models.iterrows():
        values = [row['Performance'], row['Speed'], row['Compactness']]
        values += values[:1]  # Fermer le polygone
        
        # Tracer le polygone
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Modèle'])
        ax.fill(angles, values, alpha=0.1)
    
    # Ajouter les étiquettes
    plt.xticks(angles[:-1], categories)
    
    # Ajouter une légende
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Comparaison Radar des 5 Meilleurs Modèles', size=15)
    plt.savefig('visualisations/radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_attention(model, tokenizer, text):
    """
    Visualise l'attention du modèle sur un texte donné.
    
    Args:
        model: Modèle entraîné (BERT ou DistilBERT)
        tokenizer: Tokenizer utilisé pour le modèle
        text: Texte à analyser
    """
    import numpy as np
    
    # Tokenisation
    inputs = tokenizer(text, return_tensors="tf")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].numpy())
    
    # Obtenir les sorties d'attention
    attention = None
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        outputs = model(inputs, output_attentions=True)
        attention = outputs.attentions
        # Prendre la dernière couche d'attention
        if attention:
            attention_layer = attention[-1].numpy()
    elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        outputs = model(inputs, output_attentions=True)
        attention = outputs.attentions
        # Prendre la dernière couche d'attention
        if attention:
            attention_layer = attention[-1].numpy()
    
    if attention is None:
        print("Impossible d'extraire l'attention de ce modèle")
        return
    
    # Visualiser l'attention
    plt.figure(figsize=(10, 8))
    
    # Pour simplifier, on utilise la première tête d'attention
    head_idx = 0
    att_matrix = attention_layer[0, head_idx]
    
    # Créer la heatmap
    plt.imshow(att_matrix, cmap='viridis')
    
    # Ajouter les labels des tokens
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.title(f"Matrice d'attention", fontsize=14)
    plt.tight_layout()
    plt.savefig('visualisations/attention_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualiser l'attention sur un token spécifique
    plt.figure(figsize=(12, 6))
    
    # On choisit un token représentatif (qui n'est pas un token spécial)
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
    content_token_idx = None
    for i, token in enumerate(tokens):
        if token not in special_tokens:
            content_token_idx = i
            break
    
    if content_token_idx is not None:
        # Attention de ce token vers tous les autres
        token_attention = att_matrix[content_token_idx]
        
        # Créer un barplot
        plt.bar(range(len(tokens)), token_attention, color='steelblue')
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.title(f"Attention du token '{tokens[content_token_idx]}' vers les autres tokens", fontsize=14)
        plt.tight_layout()
        plt.savefig('visualisations/token_attention.png', dpi=300, bbox_inches='tight')
        plt.show()