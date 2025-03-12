# Importation des bibliothèques nécessaires
import numpy as np  # numpy est utilisé pour manipuler des tableaux et effectuer des calculs mathématiques.
import joblib  # joblib est utilisé pour charger des objets sauvegardés, comme des modèles ou des scalers.
from tensorflow.keras.models import load_model  # load_model permet de charger un modèle Keras sauvegardé.
from data_preprocessor import DataPreprocessor  # Importation du préprocesseur de données pour traiter les données avant de générer un signal.
from config import MODEL_PATH  # Importation du chemin vers le modèle à partir du fichier de configuration.
from logger import setup_logger  # Importation de la fonction pour configurer le logger (suivi des événements).
import os  # os est utilisé pour interagir avec le système d'exploitation, ici utilisé pour vérifier les fichiers.

# Configuration du logger pour enregistrer les événements et erreurs
logger = setup_logger(__name__)

# Initialisation du préprocesseur de données pour préparer les données brutes.
preprocessor = DataPreprocessor()

# Chargement du scaler pour normaliser les nouvelles données à l'aide des mêmes paramètres que lors de l'entraînement.
try:
    scaler = joblib.load('models/scaler.joblib')  # Chargement du scaler depuis un fichier sauvegardé.
    preprocessor.scaler = scaler  # Assignation du scaler au préprocesseur pour l'utiliser dans le traitement des données.
except Exception as e:
    logger.error(f"Erreur lors du chargement du scaler : {e}")  # Enregistre une erreur si le chargement du scaler échoue.
    preprocessor.scaler = None  # Définit le scaler comme None si une erreur survient.

# Fonction pour déterminer le type de modèle et charger le modèle sauvegardé.
def load_best_model():
    if MODEL_PATH.endswith('.h5'):  # Si le fichier du modèle a une extension '.h5', il s'agit d'un modèle Keras.
        model = load_model(MODEL_PATH)  # Chargement du modèle Keras.
        model_type = 'keras'  # Indication que c'est un modèle Keras.
    elif MODEL_PATH.endswith('.joblib'):  # Si le fichier a une extension '.joblib', il s'agit d'un modèle scikit-learn ou XGBoost.
        model = joblib.load(MODEL_PATH)  # Chargement du modèle scikit-learn ou XGBoost.
        model_type = 'sklearn'  # Indication que c'est un modèle scikit-learn.
    else:
        logger.error("Type de modèle non supporté.")  # Enregistre une erreur si le type de fichier du modèle n'est pas reconnu.
        model = None  # Aucun modèle n'est chargé si le type est incorrect.
        model_type = None  # Le type de modèle est défini comme None.
    return model, model_type  # Retourne le modèle chargé et son type.

# Chargement du meilleur modèle en utilisant la fonction ci-dessus.
model, model_type = load_best_model()

# Fonction pour générer un signal d'achat, de vente ou de maintien en fonction des données passées.
def generate_signal(data):
    try:
        # Prétraitement des données avant de les utiliser pour générer un signal.
        scaled_data = preprocessor.preprocess_data(data)  # Les données sont normalisées et les indicateurs techniques sont calculés.
        if scaled_data is None ou len(scaled_data) < 60:  # Vérifie si les données sont suffisantes pour générer un signal.
            logger.error("Données insuffisantes pour générer un signal.")  # Enregistre une erreur si les données sont insuffisantes.
            return 'HOLD'  # Retourne un signal "HOLD" si les données sont insuffisantes.

        # Prépare l'entrée pour le modèle en prenant les 60 dernières périodes.
        X_input = np.array([scaled_data[-60:]])

        # Si le modèle est de type Keras
        if model_type == 'keras':
            prediction = model.predict(X_input)  # Effectue une prédiction avec le modèle Keras.
            probability = prediction[0][0]  # Récupère la probabilité de la classe "achat" (1).

        # Si le modèle est de type scikit-learn
        elif model_type == 'sklearn':
            # Remodelage des données pour qu'elles soient compatibles avec les modèles scikit-learn.
            X_input = X_input.reshape(1, -1)
            probability = model.predict_proba(X_input)[0][1]  # Récupère la probabilité pour la classe "achat" (1).

        else:
            logger.error("Aucun modèle valide n'est chargé.")  # Enregistre une erreur si aucun modèle valide n'est chargé.
            return 'HOLD'  # Retourne un signal "HOLD" si aucun modèle valide n'est chargé.

        # Ajustement dynamique des seuils pour déclencher un achat ou une vente
        buy_threshold = 0.6  # Seuil pour déclencher un signal d'achat.
        sell_threshold = 0.4  # Seuil pour déclencher un signal de vente.

        # Si la probabilité dépasse le seuil d'achat, le signal est "BUY".
        if probability > buy_threshold:
            logger.info(f"Signal d'achat détecté avec une probabilité de {probability:.2f}.")  # Log le signal d'achat.
            return 'BUY'  # Retourne un signal "BUY".

        # Si la probabilité est en dessous du seuil de vente, le signal est "SELL".
        elif probability < sell_threshold:
            logger.info(f"Signal de vente détecté avec une probabilité de {probability:.2f}.")  # Log le signal de vente.
            return 'SELL'  # Retourne un signal "SELL".

        # Si aucune condition n'est remplie, le signal est "HOLD" (pas d'action).
        else:
            logger.info(f"Aucun signal clair détecté. Probabilité de {probability:.2f}.")  # Log l'absence de signal clair.
            return 'HOLD'  # Retourne un signal "HOLD".

    except Exception as e:
        # Si une erreur se produit pendant la génération du signal, elle est enregistrée ici.
        logger.error(f"Erreur lors de la génération du signal : {e}")
        return 'HOLD'  # Retourne "HOLD" en cas d'erreur.
