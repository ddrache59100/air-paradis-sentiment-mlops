import unittest
import json
from api import app, clean_text, predict_with_model, load_model_config

class TestAPIFunctions(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_clean_text(self):
        # Test de nettoyage de texte basique
        text = "Hello @user! Check out https://example.com #example"
        expected = "hello check out example"
        self.assertEqual(clean_text(text), expected)
        
        # Test avec texte vide
        self.assertEqual(clean_text(""), "")
        
        # Test avec input non-texte
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_predict_endpoint(self):
        # Test avec un tweet positif
        response = self.app.post('/predict', 
                               data=json.dumps({'text': 'I love this airline!'}),
                               content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('label', data)
        self.assertIn('probabilities', data)
        
        # Test avec requête mal formée
        response = self.app.post('/predict', 
                               data=json.dumps({'wrong_field': 'text'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)

    # Nouveau test 1: Tester l'endpoint status
    def test_status_endpoint(self):
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertTrue(data['model_loaded'])  # Vérifie que le modèle est chargé
    
    # Nouveau test 2: Tester l'endpoint racine
    def test_root_endpoint(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('name', data)
        self.assertIn('version', data)
        self.assertIn('endpoints', data)
    
    # Nouveau test 3: Tester le comportement de prédiction avec différents types de texte
    def test_prediction_with_different_inputs(self):
        # Test avec un tweet clairement positif
        positive_text = "I absolutely love flying with Air Paradis! Best experience ever!"
        response = self.app.post('/predict', 
                              data=json.dumps({'text': positive_text}),
                              content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(data['label'], "Positif")  # Doit prédire positif
        
        # Test avec un tweet clairement négatif
        negative_text = "Worst airline ever! Terrible service and always delayed."
        response = self.app.post('/predict', 
                              data=json.dumps({'text': negative_text}),
                              content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(data['label'], "Negatif")  # Doit prédire négatif
        
        # Test avec un texte vide
        response = self.app.post('/predict', 
                              data=json.dumps({'text': ""}),
                              content_type='application/json')
        self.assertEqual(response.status_code, 400)  # Devrait retourner une erreur 400

if __name__ == '__main__':
    unittest.main()