# model_training.py 

# Imports nécessaires
import pickle
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Transformers (si utilisé)
try:
    from transformers import TFBertForSequenceClassification, BertTokenizer
    from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
    from transformers import AdamWeightDecay
    transformers_available = True
except ImportError:
    transformers_available = False

# MLflow
import mlflow

# Modules personnalisés
from decorators import time_function
from config import CONFIG, print_md
from metrics_utils import calculate_metrics, save_model_results
from visualization_utils import plot_confusion_matrix, plot_roc_curve
from data_processing import prepare_data_for_bert, prepare_data_for_dl, apply_augmentation, load_glove_embeddings, create_random_embeddings
from model_definitions import create_cnn_model, create_lstm_model


def train_all_classical_models(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue tous les modèles classiques.
    
    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        
    Returns:
        results_df: DataFrame contenant les résultats
    """
    # Liste des modèles à tester
    models = [
        (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42), 
         "Régression Logistique"),
        (MultinomialNB(alpha=0.1), 
         "Naive Bayes"),
        (LinearSVC(C=1.0, max_iter=1000, dual=False, random_state=42),
         "SVM Linéaire"),
        (RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            n_jobs=-1,
            random_state=42), 
         "Random Forest")
    ]
    
    # Créer un tableau pour stocker les résultats
    results = []
    
    # Vectoriseur commun pour tous les modèles
    vectorizer = TfidfVectorizer(
        max_features=50000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    # Entraîner et évaluer chaque modèle
    for model, model_name in models:
        try:
            # Entraîner le modèle
            result = train_evaluate_classical_model(
                model, X_train, X_test, y_train, y_test, 
                vectorizer=vectorizer, model_name=model_name
            )
            
            # Extraire les résultats
            pipeline, metrics, _, _ = result[0]  # On ignore le temps d'exécution de la fonction
            
            # Ajouter les résultats au tableau
            results.append({
                'Modèle': model_name,
                'Type': 'Classique',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', None),
                'Temps (s)': metrics['training_time'],
                'Taille (MB)': metrics['model_size_mb'],
                'Inf. (s)': metrics['inference_time'],
                'Inf. (ms/ex)': metrics['inference_time_per_sample'],
                'Taille (MB)': metrics['model_size_mb']
            })
            
            print(f"\n{'-'*50}\n")
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle {model_name}: {e}")
    
    # Créer un DataFrame à partir des résultats
    results_df = pd.DataFrame(results)
    
    # Assurez-vous que les colonnes requises existent
    if 'Modèle' not in results_df.columns:
        # Si les résultats sont vides ou n'ont pas les bonnes colonnes
        # Créez un DataFrame minimal avec les colonnes nécessaires
        if len(results) == 0:
            results_df = pd.DataFrame(columns=['Modèle', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 
                                              'ROC AUC', 'Temps (s)', 'Taille (MB)'])
    
    # Sauvegarder les résultats
    results_path = "results/classical/comparison.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Résultats sauvegardés: {results_path}")
    
    return results_df

@time_function
def train_evaluate_classical_model(model, X_train, X_test, y_train, y_test, vectorizer=None, model_name=None):
    """
    Entraîne et évalue un modèle classique de ML.
    
    Args:
        model: Modèle sklearn à entraîner
        X_train: Données d'entraînement (textes)
        X_test: Données de test (textes)
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        vectorizer: Vectoriseur de texte (si None, utilise TfidfVectorizer)
        model_name: Nom du modèle pour le suivi
        
    Returns:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
        predictions: Prédictions sur l'ensemble de test
        predictions_proba: Probabilités de prédiction (si disponible)
    """
    # Nom par défaut si non fourni
    model_name = model_name or model.__class__.__name__

    print_md(f"""##### Entraînement du modèle: {model_name}""")
        # print(f"Entraînement du modèle: {model_name}")
    
    # Utiliser TfidfVectorizer comme vectoriseur par défaut
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=50000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )
    
    # Créer un pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', model)
    ])
    
    # Entraîner le modèle
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Mesurer le temps d'inférence
    start_time = time.time()
    # Faire plusieurs passes pour obtenir une mesure plus stable
    n_repeats = 5
    for _ in range(n_repeats):
        y_pred = pipeline.predict(X_test)
    inference_time = (time.time() - start_time) / n_repeats
    inference_time_per_sample = inference_time / len(X_test)
    
    # Probabilités (si disponible)
    y_pred_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive
        except:
            y_pred_proba = None
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['training_time'] = training_time
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_sample'] = inference_time_per_sample * 1000  # en millisecondes
    
    # Afficher les métriques
    print(f"\nRésultats pour {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Temps d'entraînement: {training_time:.2f} secondes")
    print(f"Temps d'inférence total: {inference_time:.4f} secondes")
    print(f"Temps d'inférence par exemple: {inference_time_per_sample * 1000:.4f} ms")
    
    # Calculer la taille du modèle en mégaoctets
    
    model_size_bytes = len(pickle.dumps(pipeline))
    model_size_mb = model_size_bytes / (1024 * 1024)
    metrics['model_size_mb'] = model_size_mb
    print(f"Taille du modèle: {model_size_mb:.2f} MB")
    
    # Sauvegarder le modèle
    model_path = f"models/classical/{model_name.replace(' ', '_').lower()}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Modèle sauvegardé: {model_path}")
    
    
    # Journaliser les résultats dans MLflow
    with mlflow.start_run(run_name=model_name):
        # Ajouter une description détaillée du modèle
        description = f"""
        Modèle de classification: {model_name}
        Type: Modèle classique de machine learning
        Vectoriseur: {type(vectorizer).__name__}
        Features: TF-IDF ou Bag of Words
        Prétraitement: {'Avec lemmatisation' if CONFIG.get('USE_LEMMATIZATION', True) else 'Sans lemmatisation'}
        Taille d'échantillon: {CONFIG.get('SAMPLE_SIZE', 'Dataset complet')}
        Date d'entraînement: {time.strftime('%Y-%m-%d %H:%M:%S')}

        Description: Ce modèle utilise l'approche classique de machine learning pour la classification de sentiments.
        Il convertit le texte en vecteurs numériques via TF-IDF ou Bag of Words, puis applique un algorithme 
        de classification pour prédire le sentiment (positif ou négatif).
        """
        mlflow.set_tag("mlflow.note.content", description)

        # Paramètres
        params = {}
        
        # Extraire les paramètres du modèle
        if hasattr(model, "get_params"):
            params.update(model.get_params())
        
        # Extraire les paramètres du vectoriseur
        if hasattr(vectorizer, "get_params"):
            vectorizer_params = {f"vec_{k}": v for k, v in vectorizer.get_params().items()}
            params.update(vectorizer_params)

        # Ajouter les paramètres de l'échantillon
        mlflow.log_param("sample_size", CONFIG.get("SAMPLE_SIZE", "Full dataset"))

        # Ajouter le paramètre de lemmatisation
        mlflow.log_param("use_lemmatization", CONFIG.get("USE_LEMMATIZATION", True))

        # Journaliser les paramètres
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except:
                pass  # Ignorer les paramètres qui ne peuvent pas être journalisés
        
        # Journaliser les métriques
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Sauvegarder les résultats et générer les visualisations
        save_model_results(
            model_name=model_name,
            model=pipeline,
            metrics=metrics,
            predictions=y_pred,
            y_true=y_test,
            predictions_proba=y_pred_proba,
            model_type="classical",
            section="3_1",
            combined_plots=True
        )

        # Enregistrer le modèle
        mlflow.sklearn.log_model(pipeline, "model")
    
    return pipeline, metrics, y_pred, y_pred_proba

def train_all_bert_models(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue tous les modèles BERT/DistilBERT.
    
    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        
    Returns:
        results_df: DataFrame contenant les résultats
    """
    
    # Liste des modèles BERT/DistilBERT à tester
    bert_models = [
        {
            'model_name': 'bert-base-uncased',
            'display_name': 'BERT Base',
            'epochs': 4,  # Inchangé ou légèrement augmenté car on a un meilleur early stopping
            'learning_rate': 1e-5,  # Réduit de 2e-5 à 1e-5
            'patience': 1,  # Réduit de 2 à 1
            'min_delta': 0.02,  # Augmenté de 0.01 à 0.02
            'weight_decay': 0.02  # Augmenté de 0.01 à 0.02
        },
        {
            'model_name': 'distilbert-base-uncased',
            'display_name': 'DistilBERT Base',
            'epochs': 4,
            'learning_rate': 3e-5,  # Réduit de 5e-5 à 3e-5
            'patience': 1,
            'min_delta': 0.02,
            'weight_decay': 0.02
        }
    ]
    
    # Tableau pour stocker les résultats
    results = []
    
    # Entraîner et évaluer chaque modèle
    for model_config in bert_models:
        try:
            # Préparer les données
            print_md(f"\n {'-'*50}")
            print_md(f"##### Préparation des données pour {model_config['display_name']}")
            print_md(f"{'-'*50}\n")
            # print(f"\n{'-'*50}")
            # print(f"Préparation des données pour {model_config['display_name']}")
            # print(f"{'-'*50}\n")
            
            data_prep_result = prepare_data_for_bert(
                X_train, X_test, y_train, y_test,
                model_name=model_config['model_name']
            )
            
            # Gérer le cas où le décorateur time_function emballe le résultat
            if isinstance(data_prep_result, tuple) and len(data_prep_result) == 2 and isinstance(data_prep_result[1], (int, float)):
                train_dataset, test_dataset, prep_y_train, prep_y_test, tokenizer = data_prep_result[0]
            else:
                train_dataset, test_dataset, prep_y_train, prep_y_test, tokenizer = data_prep_result
            
            # Entraîner le modèle
            print_md(f"\n{'-'*50}")
            print_md(f"##### Entraînement du modèle: {model_config['display_name']}")
            print_md(f"{'-'*50}\n")
            # print(f"\n{'-'*50}")
            # print(f"Entraînement du modèle: {model_config['display_name']}")
            # print(f"{'-'*50}\n")
            
            train_result = train_evaluate_bert_model(
                model_config['model_name'],
                train_dataset,
                test_dataset,
                prep_y_test,
                tokenizer=tokenizer,
                epochs=model_config['epochs'],
                learning_rate=model_config['learning_rate'],
                display_name=model_config['display_name'],
                patience=model_config['patience'],
                min_delta=model_config['min_delta'],
                weight_decay=model_config['weight_decay']
            )
            
            # Gérer le cas où le décorateur time_function emballe le résultat
            if isinstance(train_result, tuple) and len(train_result) == 2 and isinstance(train_result[1], (int, float)):
                # Le résultat est emballé par le décorateur time_function
                bert_output_tuple = train_result[0]
            else:
                # Le résultat n'est pas emballé
                bert_output_tuple = train_result
            
            model, metrics, pred, pred_proba, history = bert_output_tuple
            
            # Créer le dictionnaire de résultats de base
            result_dict = {
                'Modèle': model_config['display_name'],
                'Type': 'Transformer',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', None),
                'Temps (s)': metrics['training_time'],
                'Taille (MB)': metrics.get('model_size_mb', 0)
            }
            
            # Ajouter les métriques d'inférence si elles existent
            if 'inference_time' in metrics:
                result_dict['Inf. (s)'] = metrics['inference_time']
            
            if 'inference_time_per_sample' in metrics:
                result_dict['Inf. (ms/ex)'] = metrics['inference_time_per_sample']
            
            results.append(result_dict)
            
            print(f"\n{'-'*50}\n")
            
        except Exception as e:
            import traceback
            print(f"Erreur lors de l'entraînement du modèle {model_config['display_name']}: {e}")
            print(traceback.format_exc())
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarder les résultats
    results_path = "results/bert/comparison.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Résultats sauvegardés: {results_path}")
    
    return results_df

@time_function
def train_evaluate_bert_model(model_name, train_dataset, test_dataset, y_test, 
                           tokenizer=None, epochs=3, learning_rate=1e-5, display_name=None,
                           patience=2, min_delta=0.02, weight_decay=0.02):
    """
    Entraîne et évalue un modèle BERT/DistilBERT avec paramètres optimisés.
    
    Args:
        model_name: Nom du modèle BERT/DistilBERT (ex: 'bert-base-uncased')
        train_dataset: Dataset TensorFlow pour l'entraînement
        test_dataset: Dataset TensorFlow pour le test
        y_test: Étiquettes de test
        epochs: Nombre d'époques d'entraînement
        learning_rate: Taux d'apprentissage
        display_name: Nom d'affichage du modèle (optionnel)
        patience: Patience pour l'early stopping
        min_delta: Delta minimum pour l'early stopping
        weight_decay: Taux de décroissance des poids
        
    Returns:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
        y_pred: Prédictions sur l'ensemble de test
        y_pred_proba: Probabilités de prédiction
        history: Historique d'entraînement
    """
    import tensorflow as tf
    import numpy as np
    import time
    from transformers import TFBertForSequenceClassification, TFDistilBertForSequenceClassification, AdamWeightDecay
    import os
    import subprocess
    import mlflow
    
    display_name = display_name or model_name
    print(f"Entraînement du modèle: {display_name}")

    # Optimisation: Utilisation de la précision mixte pour accélérer l'entraînement sur GPU
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Sélection du modèle
    if 'distilbert' in model_name.lower():
        model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2, dropout=0.3)  # Réduit de 0.4 à 0.3 pour DistilBERT
    else:
        model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2, hidden_dropout_prob=0.15, attention_probs_dropout_prob=0.2)  # Réduit de 0.3 à 0.2 pour BERT
        
        # Gel des couches - uniquement pour BERT
        if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
            for i, layer in enumerate(model.bert.encoder.layer):
                if i < 4:  # Figer les 4 premières couches
                    layer.trainable = False
    
    # Estimer le nombre de steps par époque
    # Extraire la taille de batch du dataset
    for inputs, _ in train_dataset.take(1):
        batch_size = tf.shape(inputs['input_ids'])[0].numpy()
        break
    
    num_examples = sum(1 for _ in train_dataset) * batch_size  # Estimation approximative
    num_batches_per_epoch = num_examples // batch_size
    warmup_steps = int(0.1 * num_batches_per_epoch * epochs)  # 10% des steps totaux
    total_steps = num_batches_per_epoch * epochs
    
    # Créer un scheduler de learning rate avec warmup
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps - warmup_steps,
        end_learning_rate=learning_rate * 0.1,
    )
    
    # Wrapper pour ajouter le warmup
    class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, lr_schedule, warmup_steps, initial_lr):
            self.lr_schedule = lr_schedule
            self.warmup_steps = warmup_steps
            self.initial_lr = initial_lr
            
        def __call__(self, step):
            warmup_lr = self.initial_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
            return tf.cond(
                step < self.warmup_steps,
                lambda: warmup_lr,
                lambda: self.lr_schedule(step - self.warmup_steps)
            )
    
    scheduled_lr = WarmupScheduler(lr_scheduler, warmup_steps, learning_rate)
    
    # Optimiseur avec weight decay augmenté
    optimizer = AdamWeightDecay(
        learning_rate=scheduled_lr,
        weight_decay_rate=weight_decay,
        epsilon=1e-8,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )
    
    # Compilation du modèle
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=True
    )
    
    # Classe de callback pour l'early stopping
    class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
        def __init__(self, patience=patience, min_delta=min_delta):
            super(EarlyStoppingAtMinLoss, self).__init__()
            self.patience = patience
            self.min_delta = min_delta
            self.best_weights = None
            self.best = float('inf')
            self.wait = 0
            self.stopped_epoch = 0
            
        def on_train_begin(self, logs=None):
            self.best = float('inf')
            self.wait = 0
            self.stopped_epoch = 0
            self.best_weights = None
            
        def on_epoch_end(self, epoch, logs=None):
            current = logs.get('val_loss')
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
                self.best_weights = self.model.get_weights()
                print(f"\nEpoch {epoch+1}: val_loss improved to {current:.5f}")
            else:
                self.wait += 1
                print(f"\nEpoch {epoch+1}: val_loss did not improve from {self.best:.5f}")
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print(f"Epoch {epoch+1}: early stopping")
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                        print(f"Restoring model weights from the end of the best epoch: {epoch+1-self.wait}.")
    
    # Créer les callbacks
    callbacks = [
        EarlyStoppingAtMinLoss(patience=patience, min_delta=min_delta)
    ]
    
    # Entraînement du modèle avec early stopping
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    # Mesurer le temps d'inférence
    # Récupérer un lot du dataset de test pour mesurer le temps d'inférence
    test_batch = next(iter(test_dataset))
    
    start_time = time.time()
    # Faire plusieurs passes pour obtenir une mesure plus stable
    n_repeats = 5
    for _ in range(n_repeats):
        _ = model(test_batch[0], training=False)
    inference_time_batch = (time.time() - start_time) / n_repeats
    
    # Mesurer le temps d'inférence sur l'ensemble complet
    start_time = time.time()
    logits = model.predict(test_dataset)[0]
    inference_time_full = time.time() - start_time
    
    # Estimer le temps par échantillon en se basant sur le lot de test
    batch_size = len(next(iter(test_dataset))[0]['input_ids'])
    inference_time_per_sample = inference_time_batch / batch_size * 1000  # en millisecondes
    
    y_pred_proba = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    
    # Extraction de y_test à partir du dataset
    y_test_values = np.array([y.numpy() for x, y in test_dataset.unbatch()])
    
    # Calcul des métriques
    metrics = calculate_metrics(y_test_values, y_pred, y_pred_proba)
    metrics['training_time'] = training_time
    metrics['inference_time'] = inference_time_full
    metrics['inference_time_batch'] = inference_time_batch
    metrics['inference_time_per_sample'] = inference_time_per_sample
    
    # Afficher les métriques
    print(f"\nRésultats pour {display_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Temps d'entraînement: {training_time:.2f} secondes")
    print(f"Temps d'inférence total: {inference_time_full:.4f} secondes")
    print(f"Temps d'inférence par exemple: {inference_time_per_sample:.4f} ms")
    
    # Sauvegarder le modèle
    model_dir = f"models/bert/{display_name.replace(' ', '_').lower()}"
    model.save_pretrained(model_dir)
    print(f"Modèle sauvegardé: {model_dir}")
    
    # Calculer la taille du modèle en mégaoctets
    try:
        # Obtenir la taille du répertoire
        size_command = f"du -sm {model_dir}"
        result = subprocess.run(size_command, shell=True, capture_output=True, text=True)
        
        model_size_mb = float(result.stdout.split()[0])
        metrics['model_size_mb'] = model_size_mb
        print(f"Taille du modèle: {model_size_mb:.2f} MB")
    except Exception as e:
        print(f"Erreur lors du calcul de la taille du modèle: {e}")
        # Estimation approximative de la taille
        if 'distilbert' in model_name.lower():
            metrics['model_size_mb'] = 250  # Taille approximative de DistilBERT
        else:
            metrics['model_size_mb'] = 420  # Taille approximative de BERT
    
    # Sauvegarder le tokenizer
    if tokenizer is not None:
        tokenizer_path = f"{model_dir}/tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer sauvegardé: {tokenizer_path}")

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

        Description: Ce modèle utilise l'architecture Transformer pour la classification de sentiments.
        Il s'appuie sur des mécanismes d'attention pour comprendre le contexte et les relations entre 
        les mots dans le texte, permettant une analyse plus fine du sentiment. Les modèles Transformer 
        représentent l'état de l'art pour l'analyse de texte.
        """
        mlflow.set_tag("mlflow.note.content", description)

        # Paramètres
        params = {
            'model_name': model_name,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'patience': patience,
            'min_delta': min_delta,
            'weight_decay': weight_decay,
            'sample_size': CONFIG.get("SAMPLE_SIZE", "Full dataset"),  # Ajout explicite de la taille d'échantillon
            'use_lemmatization': CONFIG.get("USE_LEMMATIZATION", True)  # Ajout du paramètre de lemmatisation
        }

        # Journaliser les paramètres
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Journaliser les métriques
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Journaliser l'historique d'entraînement
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

        # Sauvegarder les résultats et générer les visualisations
        model_type = "distilbert" if 'distilbert' in model_name.lower() else "bert"
        save_model_results(
            model_name=display_name,
            model=model,
            metrics=metrics,
            predictions=y_pred,
            y_true=y_test_values,
            predictions_proba=y_pred_proba,
            model_type=model_type,
            history=history,
            section="5_3",
            combined_plots=True
        )

        # Enregistrer le tokenizer
        mlflow.log_artifact(tokenizer_path, "tokenizer")

        # # Pour BERT, si vous ne pouvez pas utiliser log_model directement,
        # # au moins enregistrer tous les fichiers du modèle comme artefacts
        # mlflow.log_artifacts(model_dir, "bert_model")       

        # Essayer d'enregistrer le modèle avec MLflow
        try:
            # Pour les modèles BERT, utiliser une approche basée sur les artefacts plutôt
            # que d'essayer de créer une signature (qui est complexe pour les transformers)
            
            # Enregistrer le dossier du modèle comme artefact
            mlflow.log_artifacts(model_dir, "bert_model")
            
            # Enregistrer les métadonnées supplémentaires
            mlflow.log_dict(
                {
                    "model_type": model_type,
                    "model_name": model_name,
                    "display_name": display_name,
                    "framework": "tensorflow",
                    "num_labels": 2
                },
                "model_info.json"
            )
            
            # Enregistrer aussi le chemin comme référence
            mlflow.log_param("model_path", model_dir)
            
        except Exception as e:
            print(f"Erreur lors de l'enregistrement du modèle dans MLflow: {e}")
            # Fallback: enregistrer simplement les artefacts
            mlflow.log_artifacts(model_dir, "bert_model")
            mlflow.log_param("model_path", model_dir)
        
        # Toujours enregistrer le chemin comme référence
        mlflow.log_param("model_path", model_dir)
    
    return model, metrics, y_pred, y_pred_proba, history

@time_function
def train_evaluate_dl_model(model, X_train, X_test, y_train, y_test, 
                          batch_size=None, epochs=None, model_name=None,
                          patience=None, min_delta=None, tokenizer=None):
    """
    Entraîne et évalue un modèle de deep learning.
    
    Args:
        model: Modèle Keras à entraîner
        X_train: Données d'entraînement
        X_test: Données de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        batch_size: Taille du batch (facultatif)
        epochs: Nombre d'époques (facultatif)
        model_name: Nom du modèle (facultatif)
        patience: Patience pour l'early stopping (facultatif)
        min_delta: Variation minimale pour l'early stopping (facultatif)
        tokenizer: Tokenizer utilisé pour le modèle (facultatif)
    """
    # Paramètres par défaut
    batch_size = batch_size or CONFIG["BATCH_SIZE"]
    epochs = epochs or CONFIG["EPOCHS"]
    model_name = model_name or model.__class__.__name__
    patience = patience or 5  # Valeur par défaut améliorée
    min_delta = min_delta or 0.003  # Valeur par défaut améliorée
    
    print(f"Entraînement du modèle: {model_name}")
    
    # Diviser les données d'entraînement pour obtenir un ensemble de validation
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,  # Augmentation de la taille de validation
        random_state=CONFIG["RANDOM_SEED"]
    )

    # Obtenir les callbacks avec patience et min_delta personnalisés
    callbacks = get_callbacks(model_name, patience=patience, min_delta=min_delta)
    
    # Entraîner le modèle
    start_time = time.time()
    history = model.fit(
        X_train_val, y_train_val,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Reste de la fonction inchangé...
    training_time = time.time() - start_time
    
    # Évaluer le modèle
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Mesurer le temps d'inférence
    start_time = time.time()
    # Faire plusieurs passes pour obtenir une mesure plus stable
    n_repeats = 5
    for _ in range(n_repeats):
        y_pred_proba = model.predict(X_test, verbose=0)
    inference_time = (time.time() - start_time) / n_repeats
    inference_time_per_sample = inference_time / len(X_test)
    
    # Convertir les probabilités en prédictions binaires
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['training_time'] = training_time
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_accuracy
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_sample'] = inference_time_per_sample * 1000  # en millisecondes
    
    # Afficher les métriques
    print(f"\nRésultats pour {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Temps d'entraînement: {training_time:.2f} secondes")
    print(f"Temps d'inférence total: {inference_time:.4f} secondes")
    print(f"Temps d'inférence par exemple: {inference_time_per_sample * 1000:.4f} ms")
    
    # Calculer la taille du modèle en mégaoctets
    try:
        model_path = f"models/deeplearning/{model_name.replace(' ', '_').lower()}.keras"
        model.save(model_path)
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        metrics['model_size_mb'] = model_size_mb
        print(f"Taille du modèle: {model_size_mb:.2f} MB")
    except Exception as e:
        print(f"Erreur lors du calcul de la taille du modèle: {e}")
        metrics['model_size_mb'] = None
    
    
    # tokenizer_path = f"models/deeplearning/tokenizer_{model_name.replace(' ', '_').lower()}.pkl"
    # with open(tokenizer_path, 'wb') as f:
    #     pickle.dump(tokenizer, f)

    # Sauvegarder le tokenizer si fourni
    if tokenizer is not None:
        import pickle  # S'assurer que pickle est importé ici
        tokenizer_path = f"models/deeplearning/{model_name.replace(' ', '_').lower()}_tokenizer.pkl"
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer sauvegardé pour le modèle {model_name}: {tokenizer_path}")

    # Journaliser les résultats dans MLflow
    with mlflow.start_run(run_name=model_name):
        # Ajouter une description détaillée du modèle
        description = f"""
        Modèle de classification: {model_name}
        Type: Deep Learning
        Architecture: {'CNN' if 'CNN' in model_name else 'LSTM'}
        Embeddings: {'GloVe pré-entraînés' if 'GloVe' in model_name else 'Entraînables'}
        Dimensions d'embedding: {CONFIG['EMBEDDING_DIM']}
        Longueur de séquence: {CONFIG['MAX_SEQUENCE_LENGTH']}
        Prétraitement: {'Avec lemmatisation' if CONFIG.get('USE_LEMMATIZATION', True) else 'Sans lemmatisation'}
        Taille d'échantillon: {CONFIG.get('SAMPLE_SIZE', 'Dataset complet')}
        Date d'entraînement: {time.strftime('%Y-%m-%d %H:%M:%S')}

        Description: Ce modèle utilise une approche de deep learning pour la classification de sentiments.
        Il convertit les mots en vecteurs d'embedding (représentations numériques denses), puis utilise des 
        couches {('CNN' if 'CNN' in model_name else 'LSTM')} pour capturer les motifs dans le texte et prédire 
        le sentiment (positif ou négatif).
        """
        mlflow.set_tag("mlflow.note.content", description)

        # Paramètres
        params = {
            'batch_size': batch_size,
            'epochs': epochs,
            'max_features': CONFIG["MAX_FEATURES"],
            'max_sequence_length': CONFIG["MAX_SEQUENCE_LENGTH"],
            'embedding_dim': CONFIG["EMBEDDING_DIM"],
            'sample_size': CONFIG.get("SAMPLE_SIZE", "Full dataset"),
            'use_lemmatization': CONFIG.get("USE_LEMMATIZATION", True)  # Ajout du paramètre de lemmatisation
        }
        
        # Journaliser les paramètres
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Journaliser les métriques
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Journaliser l'historique d'entraînement
        for i, (loss, val_loss, acc, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['val_loss'],
            history.history['accuracy'],
            history.history['val_accuracy']
        )):
            mlflow.log_metric('loss', loss, step=i)
            mlflow.log_metric('val_loss', val_loss, step=i)
            mlflow.log_metric('accuracy', acc, step=i)
            mlflow.log_metric('val_accuracy', val_acc, step=i)
        
        # Sauvegarder les résultats et générer les visualisations
        save_model_results(
            model_name=model_name,
            model=model,
            metrics=metrics,
            predictions=y_pred,
            y_true=y_test,
            predictions_proba=y_pred_proba,
            model_type="deeplearning",
            history=history,
            section="4_3",
            combined_plots=True
        )

        # Enregistrer le tokenizer comme artefact
        if tokenizer is not None:
            mlflow.log_artifact(tokenizer_path, "tokenizer")

        # # Enregistrer le modèle
        # mlflow.keras.log_model(model, "model")
        # Enregistrer le modèle avec signature pour éviter les avertissements
        try:
            # Créer une signature avec des exemples d'entrée et de sortie en utilisant une approche générique
            from mlflow.models.signature import infer_signature, ModelSignature
            from mlflow.types.schema import Schema, TensorSpec
            
            # Méthode simplifiée qui fonctionne sans connaître la structure exacte
            if isinstance(X_test, np.ndarray):
                input_example = X_test[:5]  # Prendre quelques exemples comme échantillons
                output_example = y_pred_proba[:5]
                
                # Essayer d'inférer la signature
                signature = infer_signature(input_example, output_example)
            else:
                # Si l'inférence automatique ne fonctionne pas, créer une signature générique
                input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, X_test.shape[1]))])
                output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                
            mlflow.keras.log_model(model, "model", signature=signature)
        except Exception as e:
            print(f"Impossible de créer la signature du modèle: {e}")
            # Fallback sans signature
            mlflow.keras.log_model(model, "model")
    
    return model, metrics, y_pred, y_pred_proba, history


def train_all_dl_models(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue tous les modèles de deep learning.
    
    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        
    Returns:
        results_df: DataFrame contenant les résultats
        tokenizer: Tokenizer utilisé pour la préparation des données
        word_index: Dictionnaire mapping mot -> index 
        vocab_size: Taille du vocabulaire
    """
    # Préparer les données pour les modèles DL
    result = prepare_data_for_dl(X_train, X_test, y_train, y_test)
    X_train_seq, X_test_seq, y_train, y_test, tokenizer, word_index, vocab_size = result[0]
    
    # Appliquer l'augmentation de données ici
    X_train_seq_aug, y_train_aug = apply_augmentation(X_train_seq, y_train, ratio=0.2)
    print(f"Données d'entraînement augmentées: {len(X_train_seq)} → {len(X_train_seq_aug)} séquences")
    
    # Utiliser les données augmentées pour la suite
    X_train_seq = X_train_seq_aug
    y_train = y_train_aug
    
    # Charger les embeddings GloVe
    result = load_glove_embeddings(word_index, embedding_dim=CONFIG["EMBEDDING_DIM"], max_features=CONFIG["MAX_FEATURES"])
    embedding_matrix, glove_vocab_size = result[0]
    
    # Liste des modèles à entraîner
    models = [
        {
            'create_func': create_cnn_model,
            'name': 'CNN (embeddings entraînables)',
            'params': {
                'vocab_size': vocab_size,
                'embedding_dim': CONFIG["EMBEDDING_DIM"],
                'max_sequence_length': CONFIG["MAX_SEQUENCE_LENGTH"],
                'embedding_matrix': None  # Pas d'embeddings pré-entraînés
            }
        },
        {
            'create_func': create_cnn_model,
            'name': 'CNN (GloVe)',
            'params': {
                'vocab_size': glove_vocab_size,  # Utiliser la taille du vocabulaire pour GloVe
                'embedding_dim': CONFIG["EMBEDDING_DIM"],
                'max_sequence_length': CONFIG["MAX_SEQUENCE_LENGTH"],
                'embedding_matrix': embedding_matrix
            }
        },
        {
            'create_func': create_lstm_model,
            'name': 'LSTM (embeddings entraînables)',
            'params': {
                'vocab_size': vocab_size,
                'embedding_dim': CONFIG["EMBEDDING_DIM"],
                'max_sequence_length': CONFIG["MAX_SEQUENCE_LENGTH"],
                'embedding_matrix': None
            },
            'training_params': {
                'batch_size': 32,        # Batch size réduit
                'epochs': 15,            # Plus d'époques potentielles
                'patience': 6,           # Plus de patience pour l'early stopping
                'validation_size': 0.15, # Augmentation de la taille de validation
                'min_delta': 0.003       # Seuil de delta pour early stopping
            }
        },
        {
            'create_func': create_lstm_model,
            'name': 'LSTM (GloVe)',
            'params': {
                'vocab_size': glove_vocab_size,
                'embedding_dim': CONFIG["EMBEDDING_DIM"],
                'max_sequence_length': CONFIG["MAX_SEQUENCE_LENGTH"],
                'embedding_matrix': embedding_matrix
            },
            'training_params': {
                'batch_size': 32,
                'epochs': 15,
                'patience': 6,
                'validation_size': 0.15,
                'min_delta': 0.003
            }
        }
    ]
        
    # Créer un tableau pour stocker les résultats
    results = []
    
    # Entraîner et évaluer chaque modèle
    for model_config in models:
        try:
            # Créer le modèle
            model = model_config['create_func'](**model_config['params'])
            model.summary()
            
            # Récupérer les paramètres d'entraînement spécifiques (s'ils existent)
            training_params = model_config.get('training_params', {})
            batch_size = training_params.get('batch_size', CONFIG["BATCH_SIZE"])
            epochs = training_params.get('epochs', CONFIG["EPOCHS"])
            patience = training_params.get('patience', 3)
            min_delta = training_params.get('min_delta', 0.001)
            
            # Entraîner le modèle
            print_md(f"\n{'-'*50}")
            print_md(f"##### Entraînement du modèle: {model_config['name']}")
            print_md(f"{'-'*50}\n")
            print(f"\n{'-'*50}")
            print(f"Entraînement du modèle: {model_config['name']}")
            print(f"{'-'*50}\n")
            
            result = train_evaluate_dl_model(
                model, X_train_seq, X_test_seq, y_train, y_test,
                model_name=model_config['name'],
                batch_size=batch_size,
                epochs=epochs,
                tokenizer=tokenizer,
                patience=patience,
                min_delta=min_delta
            )
            
            # Extraire les résultats
            model, metrics, _, _, _ = result[0]  # On ignore le temps d'exécution de la fonction
            
            # Ajouter les résultats au tableau
            results.append({
                'Modèle': model_config['name'],
                'Type': 'Deep Learning',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', None),
                'Temps (s)': metrics['training_time'],
                'Inf. (s)': metrics['inference_time'],
                'Inf. (ms/ex)': metrics['inference_time_per_sample'],
                'Taille (MB)': metrics['model_size_mb'] if metrics['model_size_mb'] is not None else 0
            })
            
            print(f"\n{'-'*50}\n")
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle {model_config['name']}: {e}")
    
    # Créer un DataFrame à partir des résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarder les résultats
    results_path = "results/deeplearning/comparison.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Résultats sauvegardés: {results_path}")
    
    return results_df, tokenizer, word_index, vocab_size

# ================================
# Callbacks spécifiques pour l'entraînement de modèles deep learning
# ================================

def get_callbacks(model_name, patience=6, min_delta=0.002):
    """
    Crée un ensemble amélioré de callbacks pour l'entraînement de modèles Keras.
    
    Args:
        model_name: Nom du modèle (utilisé pour le checkpoint)
        patience: Nombre d'époques à attendre avant l'early stopping
        min_delta: Variation minimale pour considérer une amélioration
        
    Returns:
        Liste de callbacks
    """
    checkpoint_path = f"models/checkpoints/{model_name}.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        # Early stopping avec plus de patience
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        
        # Réduction du learning rate plus progressive
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Réduction moins agressive
            patience=patience // 2,
            verbose=1,
            min_delta=min_delta,
            min_lr=1e-6,
            mode='min'
        ),
        
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
    ]
    
    return callbacks