# data_processing.py

# Imports nécessaires
import re
import os
import numpy as np
import pandas as pd
import pickle
import time
import multiprocessing
from joblib import Parallel, delayed
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

from decorators import time_function
from config import CONFIG

@time_function
def load_sentiment140_data(file_path, verbose=True):
    """
    Charge le dataset Sentiment140.
    
    Args:
        file_path: Chemin vers le fichier de données
        verbose: Si True, affiche des informations sur le chargement
        
    Returns:
        DataFrame contenant les données
    """
    if verbose:
        print(f"Chargement du dataset depuis {file_path}...")
    
    # Le dataset Sentiment140 a les colonnes suivantes:
    # target (0 = négatif, 4 = positif), 
    # id, date, flag (toujours NULL), user, text
    df = pd.read_csv(file_path, 
                     encoding='latin-1',
                     names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
    
    # Conversion du sentiment (0 = négatif, 4 = positif) en (0 = négatif, 1 = positif)
    df['sentiment'] = df['sentiment'].replace(4, 1)
    
    return df

@time_function
def preprocess_data(df, text_column='text', sentiment_column='sentiment', 
                   test_size=0.2, random_state=42, sample_size=None, 
                   lemmatize=True, n_jobs=-1):
    """
    Prétraite les données pour l'entraînement des modèles.
    
    Args:
        df: DataFrame contenant les données
        text_column: Nom de la colonne contenant le texte
        sentiment_column: Nom de la colonne contenant le sentiment
        test_size: Proportion de données pour le test
        random_state: Graine aléatoire pour la reproductibilité
        sample_size: Taille de l'échantillon à utiliser (None = tout utiliser
        lemmatize: Si True, applique la lemmatisation: 
        n_jobs: Nombre de processeurs à utiliser (-1 pour tous) )
        
    Returns:
        X_train, X_test, y_train, y_test: Données d'entraînement et de test
    """
    print("Prétraitement des données...")
    
    # Afficher la distribution originale
    print("\nDistribution originale des sentiments:")
    original_distribution = df[sentiment_column].value_counts(normalize=True)
    print(original_distribution)
    
    # Nettoyer les textes
    print("Nettoyage des textes...")
    if lemmatize:
        # Déterminer le nombre de cœurs
        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        # Nettoyer les textes en parallèle
        print(f"Nettoyage des textes en parallèle sur {num_cores} cœurs...")
        
        # Convertir la série en liste pour le traitement parallèle
        texts = df[text_column].tolist()
        
        # Traitement parallèle
        cleaned_texts = Parallel(n_jobs=num_cores)(
            delayed(clean_and_lemmatize_text)(text, lemmatize=lemmatize) for text in texts
        )
        # Ajouter les résultats au DataFrame
        df['clean_text'] = cleaned_texts
    else:
        # Ajouter les résultats au DataFrame
        df['clean_text'] = df[text_column].apply(clean_text)

    # # Ajouter les résultats au DataFrame
    # df['clean_text'] = cleaned_texts
    
    # Sous-échantillonnage si demandé
    if sample_size is not None and sample_size < len(df):
        print(f"Sous-échantillonnage à {sample_size} observations...")
        # Échantillonnage stratifié manuel pour maintenir la distribution des classes
        # Prendre un échantillon séparé pour chaque classe
        sample_dfs = []
        for sentiment_value in df[sentiment_column].unique():
            class_df = df[df[sentiment_column] == sentiment_value]
            class_size = int(sample_size * len(class_df) / len(df))
            sampled_class_df = class_df.sample(n=min(class_size, len(class_df)), random_state=random_state)
            sample_dfs.append(sampled_class_df)
        
        # Concaténer les échantillons des différentes classes
        df_sampled = pd.concat(sample_dfs)
        
        # Vérifier la distribution dans l'échantillon
        print("\nDistribution des sentiments après échantillonnage:")
        sampled_distribution = df_sampled[sentiment_column].value_counts(normalize=True)
        print(sampled_distribution)
        
        # Calculer la différence relative entre les distributions
        diff = {}
        for sentiment in original_distribution.index:
            orig = original_distribution[sentiment]
            samp = sampled_distribution[sentiment]
            rel_diff = (samp - orig) / orig * 100
            diff[sentiment] = rel_diff
        
        print("\nDifférence relative entre distributions (%):")
        for sentiment, rel_diff in diff.items():
            print(f"Sentiment {sentiment}: {rel_diff:.2f}%")
            
        df = df_sampled
        print(f"Taille de l'échantillon: {len(df)}")
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df[sentiment_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df[sentiment_column]  # Pour maintenir la même distribution
    )
    
    print(f"Taille de l'ensemble d'entraînement: {len(X_train)}")
    print(f"Taille de l'ensemble de test: {len(X_test)}")
    
    # Vérifier la distribution dans les ensembles d'entraînement et de test
    print("\nDistribution des sentiments dans l'ensemble d'entraînement:")
    print(y_train.value_counts(normalize=True))
    
    print("\nDistribution des sentiments dans l'ensemble de test:")
    print(y_test.value_counts(normalize=True))
   
    return X_train, X_test, y_train, y_test

# Sauvegarder les données prétraitées pour pouvoir les réutiliser
def save_processed_data(X_train, X_test, y_train, y_test, folder='data/processed'):
    """
    Sauvegarde les données prétraitées pour pouvoir les réutiliser.
    """
    os.makedirs(folder, exist_ok=True)
    
    with open(f'{folder}/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(f'{folder}/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    
    with open(f'{folder}/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(f'{folder}/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    print(f"Données prétraitées sauvegardées dans {folder}")


# Fonction pour charger les données prétraitées
def load_processed_data(folder='data/processed'):
    """
    Charge les données prétraitées.
    """
    with open(f'{folder}/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    
    with open(f'{folder}/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    with open(f'{folder}/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    with open(f'{folder}/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    return X_train, X_test, y_train, y_test


def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les URLs, mentions, hashtags et caractères spéciaux.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
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
    
    # Suppression des caractères non alphanumériques
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Télécharger les ressources nécessaires (à exécuter une seule fois)
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Fonction pour obtenir la partie du discours au format WordNet
def get_wordnet_pos(word):
    """Convertit le tag POS en un format compatible avec WordNet"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Par défaut: NOUN

# Fonction de lemmatisation
def lemmatize_text(text):
    """
    Lemmatise un texte en anglais.
    
    Args:
        text: Texte à lemmatiser
        
    Returns:
        Texte lemmatisé
    """
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words])

# Version améliorée de clean_text avec lemmatisation
def clean_and_lemmatize_text(text: str, lemmatize: bool = True) -> str:
    """
    Nettoie un texte et applique une lemmatisation si demandé.
    
    Args:
        text: Texte à nettoyer
        lemmatize: Si True, applique la lemmatisation
        
    Returns:
        Texte nettoyé et éventuellement lemmatisé
    """
    # Appliquer d'abord le nettoyage de base
    cleaned_text = clean_text(text)
    
    # Appliquer la lemmatisation si demandé
    if lemmatize:
        cleaned_text = lemmatize_text(cleaned_text)
    
    return cleaned_text

# Fonction d'augmentation de texte
def augment_text(text: str, prob: float = 0.3) -> str:
    """
    Applique des techniques simples d'augmentation de données au texte.
    
    Args:
        text: Texte à augmenter
        prob: Probabilité d'appliquer chaque augmentation
        
    Returns:
        Texte augmenté
    """
    import random
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    
    words = text.split()
    if len(words) <= 1:
        return text
    
    # 1. Suppression aléatoire de mots
    if random.random() < prob:
        words = [w for w in words if random.random() > 0.1]
    
    # 2. Permutation aléatoire de mots
    if random.random() < prob:
        random.shuffle(words)
    
    # 3. Duplication de mots aléatoires
    if random.random() < prob:
        word_to_duplicate = random.choice(words)
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, word_to_duplicate)
    
    return ' '.join(words)

# À ajouter dans utils.py dans une section "Data Augmentation Functions"

def apply_augmentation(X_train_seq, y_train, ratio=0.2):
    """
    Applique une augmentation de données simple aux séquences d'entraînement.
    
    Args:
        X_train_seq: Séquences d'entraînement
        y_train: Étiquettes d'entraînement
        ratio: Proportion des données à augmenter
        
    Returns:
        X_combined: Séquences originales + augmentées
        y_combined: Étiquettes correspondantes
    """
    # Nombre d'exemples à augmenter
    n_aug = int(len(X_train_seq) * ratio)
    
    # Convertir y_train en array numpy si c'est une série Pandas
    if isinstance(y_train, pd.Series):
        y_train_values = y_train.values
        y_train_indices = y_train.index
    else:
        y_train_values = y_train
        y_train_indices = np.arange(len(y_train))
    
    # Sélection aléatoire d'indices
    indices_pos = np.random.choice(len(X_train_seq), size=n_aug, replace=False)
    
    # Nouvelles séquences augmentées
    X_aug = []
    y_aug = []
    
    for pos in indices_pos:
        seq = X_train_seq[pos].copy()
        # Trouver les positions non nulles (tokens réels, pas le padding)
        non_zeros = seq.nonzero()[0]
        
        if len(non_zeros) > 5:  # Seulement pour les séquences suffisamment longues
            # Masquer aléatoirement 10-15% des tokens
            mask_count = max(1, int(len(non_zeros) * 0.1))
            mask_positions = np.random.choice(non_zeros, size=mask_count, replace=False)
            
            # Créer la séquence augmentée en mettant à zéro les positions masquées
            aug_seq = seq.copy()
            aug_seq[mask_positions] = 0
            
            X_aug.append(aug_seq)
            y_aug.append(y_train_values[pos])
    
    # Concaténer avec les données originales
    X_combined = np.vstack([X_train_seq, np.array(X_aug)])
    y_combined = np.concatenate([y_train_values, np.array(y_aug)])
    
    return X_combined, y_combined

def augment_text_improved(text, prob=0.3):
    """
    Technique améliorée d'augmentation de données textuelles.
    
    Args:
        text: Texte à augmenter
        prob: Probabilité d'appliquer chaque augmentation
        
    Returns:
        Texte augmenté
    """
    import random
    import nltk
    from nltk.corpus import wordnet
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    
    words = text.split()
    if len(words) <= 1:
        return text
    
    # 1. Suppression aléatoire de mots (avec probabilité réduite)
    if random.random() < prob * 0.8:  # Probabilité réduite
        words = [w for w in words if random.random() > 0.05]  # Probabilité réduite
    
    # 2. Synonyme aléatoire pour certains mots
    if random.random() < prob:
        for i in range(len(words)):
            if random.random() < 0.1:  # Seulement 10% des mots
                try:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    if len(synonyms) > 0:
                        words[i] = random.choice(synonyms)
                except:
                    pass
    
    return ' '.join(words)

from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

def get_top_words(texts, n=20, min_length=3):
    """
    Extrait les n mots les plus fréquents dans un corpus de textes.
    
    Args:
        texts: Liste ou série de textes à analyser
        n: Nombre de mots les plus fréquents à retourner
        min_length: Longueur minimale des mots à considérer
        
    Returns:
        Liste de tuples (mot, fréquence) des n mots les plus fréquents
    """
    # Télécharger les stopwords si nécessaire
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    # Nettoyer les textes et extraire les mots
    words = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Supprimer les URLs
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Supprimer les mentions
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)  # Supprimer les hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Garder seulement les lettres
        
        # Extraire les mots
        for word in text.split():
            if word not in stop_words and len(word) >= min_length:
                words.append(word)
    
    # Compter les occurrences
    word_counts = Counter(words)
    
    # Retourner les n mots les plus fréquents
    return word_counts.most_common(n)

@time_function
def prepare_data_for_dl(X_train, X_test, y_train, y_test, max_features=None, max_sequence_length=None):
    """
    Prépare les données pour les modèles de deep learning.
    
    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        max_features: Nombre maximum de mots dans le vocabulaire
        max_sequence_length: Longueur maximale des séquences
        
    Returns:
        X_train_seq: Séquences d'entraînement
        X_test_seq: Séquences de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        tokenizer: Tokenizer utilisé
        word_index: Index des mots
        vocab_size: Taille du vocabulaire
    """
    # Paramètres par défaut
    max_features = max_features or CONFIG["MAX_FEATURES"]
    max_sequence_length = max_sequence_length or CONFIG["MAX_SEQUENCE_LENGTH"]
    
    print(f"Préparation des données pour les modèles DL (vocab={max_features}, seq_len={max_sequence_length})...")
    
    # Tokenisation des textes
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    
    # Conversion des textes en séquences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding des séquences
    X_train_seq = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_test_seq = pad_sequences(X_test_seq, maxlen=max_sequence_length)
    
    # Informations sur le vocabulaire
    word_index = tokenizer.word_index
    vocab_size = min(max_features, len(word_index) + 1)  # +1 pour l'index 0 réservé au padding
    
    print(f"Taille du vocabulaire: {vocab_size}")
    print(f"Forme de X_train_seq: {X_train_seq.shape}")
    print(f"Forme de X_test_seq: {X_test_seq.shape}")
    
    # Sauvegarder le tokenizer
    tokenizer_path = "models/deeplearning/tokenizer.pkl"
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer sauvegardé: {tokenizer_path}")
    
    return X_train_seq, X_test_seq, y_train, y_test, tokenizer, word_index, vocab_size

@time_function
def prepare_data_for_bert(X_train, X_test, y_train, y_test, model_name='bert-base-uncased', max_length=128):
    """
    Prépare les données pour les modèles BERT/DistilBERT.
    
    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        model_name: Nom du modèle BERT/DistilBERT ('bert-base-uncased' ou 'distilbert-base-uncased')
        max_length: Longueur maximale des séquences
        
    Returns:
        train_dataset: Dataset TensorFlow pour l'entraînement
        test_dataset: Dataset TensorFlow pour le test
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        tokenizer: Tokenizer utilisé
    """
    # Importations nécessaires
    import tensorflow as tf
    from transformers import BertTokenizer, DistilBertTokenizer
    
    print(f"Préparation des données pour {model_name}...")
    
    # Sélection du tokenizer approprié
    if 'distilbert' in model_name.lower():
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Tokenisation
    train_encodings = tokenizer(
        list(X_train),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    test_encodings = tokenizer(
        list(X_test),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    
    # Conversion en tenseurs TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train.values
    ))
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test.values
    ))
    
    # Préparation pour l'entraînement avec optimisations
    # Augmentation de la taille du buffer de shuffle et de la taille du batch
    # Ajout de cache() et prefetch() pour optimiser les performances
    train_dataset = train_dataset.shuffle(2000).cache().batch(192).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(192).prefetch(tf.data.AUTOTUNE)
    
    print(f"Données préparées pour {model_name}")
    
    return train_dataset, test_dataset, y_train, y_test, tokenizer

@time_function
def load_glove_embeddings(word_index, embedding_dim=100, max_features=None):
    """
    Charge les embeddings GloVe et ajuste la taille de la matrice pour correspondre au vocabulaire.
    Utilise un cache pour éviter de télécharger à chaque fois.
    """
    print(f"Chargement des embeddings GloVe (dim={embedding_dim})...")
    
    # Déterminer la taille du vocabulaire
    if max_features is None:
        max_features = CONFIG["MAX_FEATURES"]
    
    # Utiliser soit max_features, soit la taille réelle du vocabulaire, selon le plus petit
    vocab_size = min(max_features, len(word_index) + 1)  # +1 pour l'index 0 réservé au padding
    print(f"Taille du vocabulaire configurée: {vocab_size}")
    
    # Chemin de cache pour la matrice d'embeddings pré-calculée
    embedding_cache_path = f"data/cache/glove_{embedding_dim}d_vocab{vocab_size}_embedding.npy"
    os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
    
    # Vérifier si la matrice d'embeddings existe déjà en cache
    if os.path.exists(embedding_cache_path):
        print(f"Chargement de la matrice d'embeddings depuis le cache: {embedding_cache_path}")
        try:
            embedding_matrix = np.load(embedding_cache_path)
            return embedding_matrix, vocab_size
        except Exception as e:
            print(f"Erreur lors du chargement du cache: {e}. Recréation de la matrice...")
    
    # Chemin vers le fichier GloVe
    glove_path = f'data/glove.6B/glove.6B.{embedding_dim}d.txt'
    
    # Vérifier si le fichier GloVe existe déjà, sinon le télécharger
    if not os.path.exists(glove_path):
        try:
            # Chemin du fichier zip
            zip_path = 'data/glove.6B.zip'
            
            # Vérifier si le zip existe déjà
            if not os.path.exists(zip_path):
                # Créer le répertoire
                os.makedirs(os.path.dirname(glove_path), exist_ok=True)
                
                # URL du fichier GloVe
                glove_url = f'https://nlp.stanford.edu/data/glove.6B.zip'
                print(f"Téléchargement de GloVe depuis {glove_url}...")
                
                # Télécharger le fichier
                import urllib.request
                urllib.request.urlretrieve(glove_url, zip_path)
                print("Téléchargement terminé.")
            else:
                print(f"Archive GloVe déjà téléchargée: {zip_path}")
            
            # Extraire les fichiers si nécessaire
            import zipfile
            print("Extraction des fichiers GloVe...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('data')
            
            print("Extraction terminée.")
            
        except Exception as e:
            print(f"Erreur lors du téléchargement/extraction des embeddings: {e}")
            return create_random_embeddings(word_index, embedding_dim, vocab_size), vocab_size
    else:
        print(f"Fichiers GloVe déjà disponibles localement: {glove_path}")
    
    # Vérifier si le fichier existe après tentative de téléchargement
    if os.path.exists(glove_path):
        # Charger les embeddings
        embeddings_index = {}  # Initialiser le dictionnaire
        try:
            print("Chargement des vecteurs GloVe...")
            with open(glove_path, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            
            print(f"GloVe: {len(embeddings_index)} mots chargés.")
            
            # Créer la matrice d'embeddings avec la taille exacte du vocabulaire
            print(f"Création de la matrice d'embeddings ({vocab_size} x {embedding_dim})...")
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            
            # Compter les mots trouvés pour statistiques
            found_words = 0
            
            # Remplir la matrice avec les embeddings
            for word, i in word_index.items():
                if i >= vocab_size:
                    continue  # Ignorer les mots hors de la taille du vocabulaire cible
                
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    found_words += 1
            
            print(f"Correspondance GloVe: {found_words}/{min(len(word_index), vocab_size-1)} mots trouvés")
            print(f"Matrice d'embeddings créée: {embedding_matrix.shape}")
            
            # Sauvegarder la matrice dans le cache
            np.save(embedding_cache_path, embedding_matrix)
            print(f"Matrice d'embeddings mise en cache: {embedding_cache_path}")
            
            return embedding_matrix, vocab_size
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings GloVe: {e}")
            return create_random_embeddings(word_index, embedding_dim, vocab_size), vocab_size
    
    # Fallback en cas d'échec: utiliser des embeddings aléatoires
    return create_random_embeddings(word_index, embedding_dim, vocab_size), vocab_size


def create_random_embeddings(word_index, embedding_dim, vocab_size):
    """
    Crée une matrice d'embeddings aléatoires.
    
    Args:
        word_index: Index des mots
        embedding_dim: Dimension des embeddings
        vocab_size: Taille du vocabulaire
        
    Returns:
        embedding_matrix: Matrice d'embeddings aléatoires
    """
    print("Création d'embeddings aléatoires...")
    
    # Créer une matrice d'embeddings aléatoires
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_matrix[0] = np.zeros(embedding_dim)  # Le vecteur de padding est à zéro
    
    # Sauvegarder la matrice
    embedding_path = f"models/deeplearning/random_{embedding_dim}d_embedding.npy"
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    np.save(embedding_path, embedding_matrix)
    print(f"Matrice d'embeddings aléatoires sauvegardée: {embedding_path}")
    
    return embedding_matrix