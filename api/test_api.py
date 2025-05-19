import requests
import json
import time
import statistics
from datetime import datetime
import os
import sys

# Prendre l'URL de base depuis les arguments de ligne de commande ou utiliser localhost par défaut
if len(sys.argv) > 1:
    base_url = sys.argv[1]
else:
    # Sinon, essayer de prendre depuis une variable d'environnement
    base_url = os.environ.get("API_URL", "http://localhost:5000")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Test de l'API sur: {base_url}")

# Test de l'endpoint status
print(f"[{datetime.now().strftime('%H:%M:%S')}] Vérification du statut de l'API...")
start_time = time.time()
response = requests.get(f"{base_url}/status")
status_time = time.time() - start_time

print("Statut de l'API:")
status_info = response.json()
print(json.dumps(status_info, indent=2))
print(f"Temps de réponse: {status_time:.4f} secondes")
print("-" * 50)

# Liste de textes à tester
test_texts = [
    "I absolutely love this airline! Best flight ever!",
    "This is the worst airline experience I've ever had.",
    "The flight was delayed by 2 hours and no compensation was offered.",
    "Air Paradis has the best customer service I've experienced.",
    "Not sure how I feel about this flight, it's just okay I guess."
]

# Collecter les temps de prédiction
prediction_times = []

# Test de l'endpoint predict
for i, text in enumerate(test_texts):
    print(f"Test {i+1}: {text}")
    
    # Mesurer le temps de prédiction
    start_time = time.time()
    response = requests.post(
        f"{base_url}/predict",
        json={"text": text}
    )
    elapsed_time = time.time() - start_time
    prediction_times.append(elapsed_time)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Sentiment: {result['label']}")
        print(f"Probabilités: Positif={result['probabilities']['Positif']:.4f}, Negatif={result['probabilities']['Negatif']:.4f}")
        print(f"Temps de prédiction: {elapsed_time:.4f} secondes")
    else:
        print(f"Erreur: {response.status_code}")
        print(response.text)
    
    print("-" * 50)

# Afficher les statistiques de performance
if prediction_times:
    print("\nStatistiques de performance:")
    print(f"Modèle testé: {status_info['model_type']} ({status_info.get('model_info', {}).get('type', 'Unknown')})")
    print(f"Chemin du modèle: {status_info['model_path']}")
    print(f"Temps moyen de prédiction: {statistics.mean(prediction_times):.4f} secondes")
    print(f"Temps médian de prédiction: {statistics.median(prediction_times):.4f} secondes")
    print(f"Temps minimum de prédiction: {min(prediction_times):.4f} secondes")
    print(f"Temps maximum de prédiction: {max(prediction_times):.4f} secondes")
    print(f"Écart type: {statistics.stdev(prediction_times):.4f} secondes si plus de 2 valeurs")