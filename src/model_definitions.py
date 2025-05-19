# model_definitions.py 

# Imports nécessaires
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Input, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def create_cnn_model(vocab_size, embedding_dim, max_sequence_length, embedding_matrix=None):
    """
    Crée un modèle CNN pour la classification de texte avec une meilleure régularisation.
    
    Args:
        vocab_size: Taille du vocabulaire
        embedding_dim: Dimension des embeddings
        max_sequence_length: Longueur maximale des séquences
        embedding_matrix: Matrice d'embeddings pré-entraînés (optionnel)
        
    Returns:
        model: Modèle Keras
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
    # Réduire la dimension des embeddings pour limiter les paramètres
    embedding_dim_reduced = min(embedding_dim, 50)
    
    model = Sequential()
    
    # Couche d'embedding avec régularisation L2
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=False  # Figer les embeddings pré-entraînés
        ))
    else:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim_reduced,  # Dimension réduite
            input_length=max_sequence_length,
            embeddings_regularizer=l2(1e-5)  # Régularisation L2 faible
        ))
    
    # Ajouter un dropout après l'embedding pour réduire l'overfitting
    model.add(Dropout(0.3))
    
    # Couches CNN avec moins de filtres et régularisation L2
    model.add(Conv1D(64, 5, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    
    # Ajouter un autre niveau de convolution avec pooling intermédiaire
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 3, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    
    # Global pooling
    model.add(GlobalMaxPooling1D())
    
    # Couches denses avec moins de neurones et plus de régularisation
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout plus élevé
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilation avec un learning rate plus faible
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(vocab_size, embedding_dim, max_sequence_length, embedding_matrix=None):
    """
    Crée un modèle LSTM pour la classification de texte avec régularisation adaptée
    au type d'embedding (entraînable ou GloVe).
    
    Args:
        vocab_size: Taille du vocabulaire
        embedding_dim: Dimension des embeddings
        max_sequence_length: Longueur maximale des séquences
        embedding_matrix: Matrice d'embeddings pré-entraînés (optionnel)
        
    Returns:
        model: Modèle Keras
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    
    model = Sequential()
    
    # Déterminer quel type de modèle nous construisons en fonction de embedding_matrix
    is_glove = embedding_matrix is not None
    
    # Configuration des hyperparamètres selon le type d'embedding
    if is_glove:
        # Paramètres optimisés pour GloVe
        lstm_units1 = 56
        lstm_units2 = 32
        dense_units = 48
        dropout_rate = 0.35
        l2_reg = 1e-5
        learning_rate = 5e-4
        
        # Couche d'embedding avec poids pré-entraînés fixés
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=False
        ))
    else:
        # Paramètres optimisés pour embeddings entraînables
        lstm_units1 = 24
        lstm_units2 = 12
        dense_units = 16
        dropout_rate = 0.5
        l2_reg = 5e-5
        learning_rate = 2e-4
        
        # Couche d'embedding entraînable avec régularisation
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
    
    # Architecture LSTM commune mais avec hyperparamètres adaptés
    model.add(Bidirectional(LSTM(lstm_units1, return_sequences=True,
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                              recurrent_regularizer=tf.keras.regularizers.l2(l2_reg))))
    model.add(Dropout(dropout_rate))
    
    model.add(Bidirectional(LSTM(lstm_units2, return_sequences=False,
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                              recurrent_regularizer=tf.keras.regularizers.l2(l2_reg))))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Ajout d'une couche intermédiaire plus petite pour le modèle avec embeddings entraînables
    if not is_glove:
        model.add(Dense(8, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dropout(0.5))
    
    # Couche dense finale
    model.add(Dense(dense_units, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(Dropout(dropout_rate + 0.1))  # Dropout plus fort avant la couche de sortie
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5),  # Ajout de clipvalue
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model