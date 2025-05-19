# api.py avec lazy loading du modèle

import os
import datetime
import re
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configuration du logger pour Application Insights
# Ajoutez ceci après l'initialisation de l'app Flask
logger = logging.getLogger(__name__)

# Importations conditionnelles selon le type de modele
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False

try:
    from transformers import TFBertForSequenceClassification, BertTokenizer
    from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False

# Vérifiez si la clé d'instrumentation est définie dans les variables d'environnement
app_insights_key = os.environ.get('APPLICATIONINSIGHTS_CONNECTION_STRING')
if app_insights_key:
    logger.addHandler(
        AzureLogHandler(
            connection_string=app_insights_key
        )
    )
    logger.setLevel(logging.INFO)
else:
    # Utiliser un logger standard si App Insights n'est pas configuré
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Initialisation de l'application Flask ---
app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Configuration pour permettre de specifier un fichier de configuration alternatif
app.config.from_mapping(
    CUSTOM_CONFIG_PATH=None  # Par defaut, aucun fichier de configuration personnalise
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.json')

# Variables globales pour le modèle et ses ressources
_model = None
_tokenizer = None
_model_type = None
_current_config = None
_model_initialized = False

def get_model():
    """Fonction pour obtenir le modèle, en le chargeant si nécessaire"""
    global _model, _tokenizer, _model_type, _current_config, _model_initialized
    if not _model_initialized:
        try:
            _model, _tokenizer, _model_type, _current_config = load_model_and_resources()
            _model_initialized = True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
    return _model, _tokenizer, _model_type, _current_config

def load_model_config():
    """
    Charge la configuration du modèle à partir du fichier de configuration.
    Utilise la configuration personnalisée si spécifiée, sinon utilise la configuration par défaut.
    """
    # Vérifier si un chemin de configuration personnalisé est spécifié
    custom_config_path = app.config.get('CUSTOM_CONFIG_PATH')
    
    # Utiliser le chemin de configuration personnalisé s'il est spécifié et existe
    if custom_config_path and os.path.exists(custom_config_path):
        config_path = custom_config_path
        print(f"Utilisation de la configuration personnalisée: {config_path}")
    else:
        config_path = DEFAULT_CONFIG_PATH
        print(f"Utilisation de la configuration par défaut: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            model_paths = json.load(f)
        
        # Résoudre les chemins relatifs par rapport au répertoire de base
        if model_paths.get("best_model") and not os.path.isabs(model_paths.get("best_model")):
            model_paths["best_model"] = os.path.join(BASE_DIR, model_paths["best_model"])
        
        if model_paths.get("tokenizer") and model_paths.get("tokenizer") is not None and not os.path.isabs(model_paths.get("tokenizer")):
            model_paths["tokenizer"] = os.path.join(BASE_DIR, model_paths["tokenizer"])
        
        return model_paths
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement de la configuration depuis {config_path}: {e}")
        # Configuration par défaut si fichier non trouvé ou invalide
        return {
            "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
            "best_model_type": "classical",
            "tokenizer": None
        }
        
# --- Fonctions de nettoyage de texte ---
def clean_text(text):
    """
    Nettoie un texte en supprimant les URLs, mentions, hashtags et caracteres speciaux.
    """
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Suppression des mentions @utilisateur
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Suppression des hashtags (on garde le mot mais pas le #)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Suppression des caracteres non alphanumeriques
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Chargement du modele et des ressources ---
def load_model_and_resources():
    """
    Charge le modele et les ressources associees en fonction du type de modele.
    """
    # Charger la configuration
    model_paths = load_model_config()
    
    best_model_path = model_paths.get("best_model")
    best_model_type = model_paths.get("best_model_type", "classical")
    tokenizer_path = model_paths.get("tokenizer")
    
    model = None
    tokenizer = None
    
    try:
        print(f"Chargement du modele {best_model_type} depuis: {best_model_path}")
        
        # Verifier que le fichier du modele existe
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Le fichier modele n'existe pas: {best_model_path}")
        
        if best_model_type == 'classical':
            try:
                with open(best_model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Modele classique charge avec succes")
                
                # Verifier si le modele a les methodes necessaires
                has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
                has_predict_proba = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
                
                print(f"Le modele a une methode 'predict': {has_predict}")
                print(f"Le modele a une methode 'predict_proba': {has_predict_proba}")
                
                if not has_predict:
                    raise ValueError("Le modele n'a pas de methode 'predict'")
                
            except (pickle.PickleError, ImportError, AttributeError) as e:
                raise RuntimeError(f"Erreur lors du chargement du modele pickle: {str(e)}. "
                                  f"Cela peut etre dû a un format incompatible ou a une version differente de scikit-learn.")
            
        elif best_model_type == 'deeplearning':
            if not tf_available:
                raise ImportError("TensorFlow n'est pas disponible, impossible de charger un modele deep learning")
            model = tf.keras.models.load_model(best_model_path)
            
            if tokenizer_path is None:
                raise ValueError("Le chemin du tokenizer est requis pour un modele deep learning")
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Le fichier tokenizer n'existe pas: {tokenizer_path}")
                
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("Modele deep learning et tokenizer charges avec succes")
            
        elif best_model_type == 'transformer':
            if not transformers_available:
                raise ImportError("La bibliotheque transformers n'est pas disponible")
                
            if 'distilbert' in best_model_path.lower():
                model = TFDistilBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                print("Modele DistilBERT et tokenizer charges avec succes")
            else:
                model = TFBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                print("Modele BERT et tokenizer charges avec succes")
                
        else:
            raise ValueError(f"Type de modele non reconnu: {best_model_type}")
            
        return model, tokenizer, best_model_type, model_paths
        
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {e}")
        import traceback
        traceback.print_exc()
        
        # Au lieu de quitter l'application, lever l'exception pour qu'elle puisse etre geree
        raise

# --- Fonction de prediction selon le type de modele ---
def predict_with_model(text, model_type, model, tokenizer=None):
    """
    Predit le sentiment d'un texte selon le type de modele.
    """
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        # Texte vide apres nettoyage
        return "Negatif", 0, {"Negatif": 1.0, "Positif": 0.0}, cleaned_text
    
    if model_type == 'classical':
        # Pour les modeles classiques (avec pipeline sklearn)
        try:
            # Prediction de la classe
            prediction_class = model.predict([cleaned_text])[0]
            
            # Verifier si le modele dispose de predict_proba
            if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
                # Utiliser predict_proba si disponible
                probas = model.predict_proba([cleaned_text])[0]
                probabilities = {"Negatif": float(probas[0]), "Positif": float(probas[1])}
            else:
                # Sinon, utiliser une approximation basee sur la classe predite
                probabilities = {
                    "Negatif": float(1.0 - prediction_class),
                    "Positif": float(prediction_class)
                }
            
            sentiment = "Positif" if prediction_class == 1 else "Negatif"
        
        except Exception as e:
            print(f"Erreur lors de la prediction avec le modele classique: {e}")
            import traceback
            traceback.print_exc()
            # Prediction par defaut en cas d'erreur
            sentiment = "Negatif"
            prediction_class = 0
            probabilities = {"Negatif": 1.0, "Positif": 0.0}
        
    elif model_type == 'deeplearning':
        # Pour les modeles deep learning
        # Tokenisation et padding
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequences, maxlen=100)  # Ajustez maxlen si necessaire
        
        # Prediction
        prediction = model.predict(padded)
        sentiment_score = float(prediction[0][0])
        prediction_class = 1 if sentiment_score > 0.5 else 0
        sentiment = "Positif" if prediction_class == 1 else "Negatif"
        probabilities = {"Negatif": float(1 - sentiment_score), "Positif": float(sentiment_score)}
        
    elif model_type == 'transformer':
        # Pour les modeles BERT/DistilBERT
        # Tokenisation
        inputs = tokenizer(cleaned_text, return_tensors="tf", padding=True, truncation=True)
        
        # Prediction
        outputs = model(inputs)
        logits = outputs.logits.numpy()
        import tensorflow as tf
        probas = tf.nn.softmax(logits, axis=1).numpy()[0]
        prediction_class = np.argmax(logits, axis=1)[0]
        sentiment = "Positif" if prediction_class == 1 else "Negatif"
        probabilities = {"Negatif": float(probas[0]), "Positif": float(probas[1])}
    
    return sentiment, int(prediction_class), probabilities, cleaned_text

@app.before_request
def log_request_info():
    print(f"Requête reçue: {request.method} {request.path}")
    if request.is_json:
        print(f"Données JSON: {request.json}")

# --- Endpoint pour changer la configuration ---
@app.route('/config', methods=['POST'])
def change_config():
    if not request.is_json:
        return jsonify({"error": "La requete doit etre au format JSON"}), 400
    
    data = request.get_json()
    config_path = data.get('config_path')
    
    if not config_path:
        return jsonify({"error": "Le chemin du fichier de configuration est requis"}), 400
    
    # Verifier si le chemin est valide
    if not os.path.exists(config_path):
        return jsonify({"error": f"Le fichier de configuration '{config_path}' n'existe pas"}), 404
    
    try:
        # Mettre à jour le chemin de configuration
        app.config['CUSTOM_CONFIG_PATH'] = config_path
        
        # Forcer le rechargement du modèle
        global _model_initialized
        _model_initialized = False
        model, tokenizer, model_type, config = get_model()
        
        return jsonify({
            "success": True,
            "message": f"Configuration modifiee avec succes: {config_path}",
            "model_type": model_type,
            "config": config
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Erreur lors du changement de configuration: {str(e)}"
        }), 500

# --- Endpoint pour verifier la configuration actuelle ---
@app.route('/config', methods=['GET'])
def get_current_config():
    model, tokenizer, model_type, current_config = get_model()
    
    return jsonify({
        "config_path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
        "is_custom": app.config.get('CUSTOM_CONFIG_PATH') is not None,
        "config": current_config,
        "model_info": {
            "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')) if model else False,
            "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')) if model else False,
            "type": str(type(model).__name__) if model else None
        }
    })

# --- Endpoint de prediction ---
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not request.is_json:
        return jsonify({"error": "La requête doit être au format JSON"}), 400

    data = request.get_json()
    text = data.get('text', None)

    if text is None or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Le champ 'text' est manquant, vide ou n'est pas une chaîne de caractères"}), 400

    try:
        # Charger le modèle si nécessaire
        model, tokenizer, model_type, current_config = get_model()
            
        # Vérifier si le modèle est chargé
        if model is None:
            return jsonify({
                "error": "Aucun modèle n'est chargé. Impossible de faire une prédiction.",
                "suggestion": "Vérifiez les logs du serveur et assurez-vous que le modèle est correctement configuré."
            }), 500
            
        # Prediction
        sentiment, prediction, probabilities, cleaned_text = predict_with_model(
            text, model_type, model, tokenizer
        )
        
        # Résultat enrichi pour le front-end
        return jsonify({
            "input_text": text,
            "cleaned_text": cleaned_text,
            "label": sentiment,
            "prediction": prediction,
            "confidence": max(probabilities.values()),  # Ajouter le niveau de confiance
            "probabilities": probabilities,
            "model_type": model_type,
            "timestamp": datetime.datetime.now().isoformat()  # Ajouter un timestamp
        })
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Une erreur interne est survenue lors de la prédiction.", 
                      "details": str(e)}), 500

# --- Endpoint de statut ---
@app.route('/status', methods=['GET'])
def status():
    # Charger le modèle si nécessaire
    model, tokenizer, model_type, current_config = get_model()
    
    return jsonify({
        "status": "API operationnelle",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None if model_type != 'classical' else True,
        "model_type": model_type,
        "model_path": current_config.get("best_model") if current_config else None,
        "config_path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
        "is_custom_config": app.config.get('CUSTOM_CONFIG_PATH') is not None,
        "model_info": {
            "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')) if model else False,
            "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')) if model else False,
            "type": str(type(model).__name__) if model else None
        }
    })

# --- Endpoint d'accueil ---
@app.route('/', methods=['GET'])
def welcome():
    """
    Page d'accueil simple pour l'API.
    """
    # Charger le modèle si nécessaire
    model, tokenizer, model_type, current_config = get_model()
    
    return jsonify({
        "name": "Air Paradis - API d'analyse de sentiment",
        "description": "Cette API permet d'analyser le sentiment de tweets pour anticiper les bad buzz.",
        "version": "1.0.0",
        "endpoints": {
            "/": "Cette page d'accueil (GET)",
            "/status": "Statut de l'API et informations sur le modele charge (GET)",
            "/predict": "Analyse le sentiment d'un texte fourni (POST)",
            "/config": "Obtenir ou modifier la configuration (GET/POST)"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/predict",
            "content_type": "application/json",
            "body": {
                "text": "Votre texte a analyser ici"
            },
            "response": {
                "label": "Positif/Negatif",
                "prediction": "Classe numerique (0/1)",
                "probabilities": "Probabilites pour chaque classe"
            }
        },
        "model_info": {
            "type": model_type,
            "path": current_config.get("best_model") if current_config else None,
            "details": {
                "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')) if model else False,
                "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')) if model else False,
                "type": str(type(model).__name__) if model else None
            }
        },
        "config_info": {
            "path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
            "is_custom": app.config.get('CUSTOM_CONFIG_PATH') is not None
        }
    })

@app.route('/ping', methods=['GET'])
def ping():
    """
    Simple endpoint pour vérifier si l'API est accessible.
    """
    return jsonify({"status": "ok", "message": "API is running"}), 200


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Endpoint pour recevoir le feedback de l'utilisateur sur une prédiction.
    """
    if not request.is_json:
        return jsonify({"error": "La requête doit être au format JSON"}), 400

    data = request.get_json()
    text = data.get('text')
    predicted_sentiment = data.get('predicted_sentiment')
    actual_sentiment = data.get('actual_sentiment')
    timestamp = data.get('timestamp', datetime.datetime.now().isoformat())

    if not all([text, predicted_sentiment, actual_sentiment]):
        return jsonify({"error": "Les champs 'text', 'predicted_sentiment' et 'actual_sentiment' sont requis"}), 400

    # Enregistrement du feedback dans Application Insights
    is_correct = predicted_sentiment == actual_sentiment
    feedback_data = {
        'event': 'prediction_feedback',
        'text': text,
        'predicted_sentiment': predicted_sentiment,
        'actual_sentiment': actual_sentiment,
        'is_correct': is_correct,
        'timestamp': timestamp
    }
    
    # Envoyer les données à Application Insights
    logger.info('prediction_feedback', extra={'custom_dimensions': feedback_data})
       
    return jsonify({
        "status": "success", 
        "message": "Feedback reçu avec succès", 
        "is_correct": is_correct
    })