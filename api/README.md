# API d'Analyse de Sentiment Air Paradis

Cette API expose le modèle CNN avec embeddings entraînables pour l'analyse de sentiment des tweets. Elle est également déployée sur Azure via GitHub Actions depuis le dépôt [air-paradis-sentiment-api-cnn-embed](https://github.com/votre-username/air-paradis-sentiment-api-cnn-embed).

## Architecture de l'API

L'API est développée avec Flask et offre:
- Chargement paresseux (lazy loading) du modèle
- Validation des entrées
- Prétraitement du texte
- Endpoints RESTful
- Monitoring via Azure Application Insights

## Endpoints

- `GET /`: Page d'accueil avec informations sur l'API
- `GET /status`: État de l'API et informations sur le modèle chargé
- `POST /predict`: Analyse le sentiment d'un texte fourni
- `POST /feedback`: Collecte le feedback sur les prédictions

## Déploiement

### Local
```bash
cd api
FLASK_APP=api.py flask run --host=0.0.0.0
```

### Production
Cette API est déployée sur Azure Web App via GitHub Actions. Le pipeline CI/CD est configuré pour:

1. Exécuter les tests unitaires
2. Construire l'application
3. Déployer sur Azure

### Tests
Des tests unitaires et d'intégration sont disponibles:


```python
python api/test_unit.py
python api/test_api.py http://localhost:5000
```
