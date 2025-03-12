# Importation des bibliothèques nécessaires pour manipuler les données et entraîner les modèles
import pandas as pd  # pandas est utilisé pour lire et manipuler des tableaux de données (DataFrames).
import numpy as np  # numpy est utilisé pour les calculs mathématiques avancés, surtout pour les tableaux (matrices).
from tensorflow.keras.models import Sequential  # Sequential permet de construire des modèles de réseau de neurones séquentiels.
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM pour les réseaux de neurones récurrents, Dense pour les couches complètement connectées, Dropout pour éviter le surapprentissage.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  # Permet d'utiliser un modèle Keras avec Scikit-learn pour faire des recherches d'hyperparamètres.
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Permet de diviser les données en ensemble de test et d'entraînement, et de faire une recherche d'hyperparamètres.
from sklearn.metrics import classification_report, confusion_matrix  # Pour générer des rapports de classification et des matrices de confusion (pour évaluer la qualité du modèle).
from sklearn.ensemble import RandomForestClassifier  # Importation du classificateur Random Forest.
from xgboost import XGBClassifier  # Importation du classificateur XGBoost.
from data_preprocessor import DataPreprocessor  # Importation du préprocesseur de données pour préparer les données avant l'entraînement.
from config import MODEL_PATH, TRAINING_DATA_PATH  # Importation des chemins pour les données d'entraînement et pour enregistrer le modèle.
from imblearn.over_sampling import SMOTE  # SMOTE est utilisé pour équilibrer les classes dans les données (quand une classe est majoritaire par rapport à une autre).
from joblib import dump  # Pour sauvegarder des objets comme des modèles ou des scalers dans des fichiers.
from logger import setup_logger  # Importation de la fonction pour configurer le système de suivi des événements (logger).

# Configuration du logger pour enregistrer les événements et erreurs
logger = setup_logger(__name__)

# Fonction pour construire le modèle LSTM (réseau de neurones récurrent)
def build_lstm_model(units=50, dropout_rate=0.2):
    model = Sequential()  # Initialisation du modèle séquentiel.
    # Ajout d'une couche LSTM avec `units` neurones, qui renvoie toutes les séquences.
    model.add(LSTM(units=units, return_sequences=True, input_shape=(60, 10)))
    # Ajout de la couche Dropout pour éviter le surapprentissage (randomisation).
    model.add(Dropout(dropout_rate))
    # Ajout d'une deuxième couche LSTM, cette fois sans renvoyer toutes les séquences.
    model.add(LSTM(units=units))
    # Ajout d'une deuxième couche Dropout.
    model.add(Dropout(dropout_rate))
    # Ajout d'une couche Dense avec une activation sigmoïde pour la classification binaire.
    model.add(Dense(1, activation='sigmoid'))
    # Compilation du modèle avec l'optimiseur Adam et la fonction de perte binaire.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model  # Retourne le modèle construit.

# Fonction pour entraîner le modèle
def train_model():
    try:
        # Chargement des données d'entraînement depuis le fichier CSV.
        data = pd.read_csv(TRAINING_DATA_PATH)
        # Initialisation du préprocesseur pour normaliser et préparer les données.
        preprocessor = DataPreprocessor()
        # Prétraitement des données (normalisation et calcul des indicateurs techniques).
        scaled_data = preprocessor.preprocess_data(data)

        # Préparation des ensembles d'entrée (X) et des cibles (y).
        X = []
        y = []
        window_size = 60  # On utilise une fenêtre de 60 périodes pour prédire la prochaine période.
        for i in range(window_size, len(scaled_data)):
            # Crée un ensemble de 60 périodes consécutives pour les features X.
            X.append(scaled_data[i-window_size:i])
            # Génération de la variable cible y (1 si le prix monte, 0 s'il baisse).
            if scaled_data[i, 0] > scaled_data[i-1, 0]:
                y.append(1)
            else:
                y.append(0)

        # Conversion des listes en tableaux numpy pour être utilisées dans l'entraînement.
        X, y = np.array(X), np.array(y)

        # Gestion des classes déséquilibrées avec SMOTE (Synthetic Minority Over-sampling Technique).
        smote = SMOTE()
        # On redimensionne X pour passer d'une structure 3D à 2D pour SMOTE.
        n_samples, time_steps, n_features = X.shape
        X_reshaped = X.reshape((n_samples, time_steps * n_features))
        # Application de SMOTE pour équilibrer les classes dans les données.
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        # Reshape des données échantillonnées pour revenir à la forme 3D.
        X_resampled = X_resampled.reshape((X_resampled.shape[0], time_steps, n_features))

        # Division des données en ensembles d'entraînement (80%) et de test (20%).
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, shuffle=False)

        # Entraînement du modèle LSTM avec recherche d'hyperparamètres via GridSearchCV.
        model = KerasClassifier(build_fn=build_lstm_model, verbose=0)  # Le modèle LSTM est encapsulé dans KerasClassifier.
        param_grid = {
            'units': [50, 100],  # Différentes tailles pour le nombre de neurones.
            'dropout_rate': [0.2, 0.3],  # Différentes valeurs pour le taux de dropout.
            'epochs': [10, 20],  # Nombre d'époques d'entraînement à tester.
            'batch_size': [32, 64]  # Différentes tailles de lots à tester.
        }
        # Recherche des meilleurs hyperparamètres avec validation croisée.
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(X_train, y_train)  # Entraînement avec les paramètres du GridSearch.

        # Affichage des meilleurs hyperparamètres trouvés.
        logger.info(f"Meilleurs hyperparamètres pour LSTM : {grid_result.best_params_}")

        # Évaluation du modèle LSTM sur l'ensemble de test.
        y_pred_lstm = grid.predict(X_test)
        report_lstm = classification_report(y_test, y_pred_lstm)  # Génère un rapport de classification.
        matrix_lstm = confusion_matrix(y_test, y_pred_lstm)  # Génère une matrice de confusion.
        logger.info("Rapport de classification LSTM :\n" + report_lstm)
        logger.info("Matrice de confusion LSTM :\n" + str(matrix_lstm))

        # Entraînement d'un modèle Random Forest.
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Entraînement avec les données aplaties (2D).
        y_pred_rf = rf_model.predict(X_test.reshape(X_test.shape[0], -1))  # Prédiction avec le modèle Random Forest.
        report_rf = classification_report(y_test, y_pred_rf)
        matrix_rf = confusion_matrix(y_test, y_pred_rf)
        logger.info("Rapport de classification Random Forest :\n" + report_rf)
        logger.info("Matrice de confusion Random Forest :\n" + str(matrix_rf))

        # Entraînement d'un modèle XGBoost.
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
        report_xgb = classification_report(y_test, y_pred_xgb)
        matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
        logger.info("Rapport de classification XGBoost :\n" + report_xgb)
        logger.info("Matrice de confusion XGBoost :\n" + str(matrix_xgb))

        # Comparaison des performances des trois modèles.
        models = {
            'LSTM': grid_result.best_score_,  # Précision du LSTM.
            'Random Forest': rf_model.score(X_test.reshape(X_test.shape[0], -1), y_test),  # Précision du Random Forest.
            'XGBoost': xgb_model.score(X_test.reshape(X_test.shape[0], -1), y_test)  # Précision du XGBoost.
        }
        best_model_name = max(models, key=models.get)  # Sélection du modèle avec la meilleure précision.
        logger.info(f"Meilleur modèle : {best_model_name} avec une précision de {models[best_model_name]}")

        # Sauvegarde du meilleur modèle.
        if best_model_name == 'LSTM':
            best_model = grid_result.best_estimator_.model  # Sauvegarde du modèle LSTM.
            best_model.save(MODEL_PATH)
        elif best_model_name == 'Random Forest':
            dump(rf_model, MODEL_PATH)  # Sauvegarde du modèle Random Forest.
        else:
            dump(xgb_model, MODEL_PATH)  # Sauvegarde du modèle XGBoost.

        # Sauvegarde du scaler pour pouvoir normaliser de nouvelles données dans le futur.
        dump(preprocessor.scaler, 'models/scaler.joblib')
        logger.info("Modèle et scaler sauvegardés avec succès.")

    except Exception as e:  # Gestion des erreurs qui peuvent survenir pendant l'entraînement.
        logger.error(f"Erreur lors de l'entraînement du modèle : {e}")

# Si le fichier est exécuté directement (pas importé comme module), la fonction train_model() est lancée.
if __name__ == "__main__":
    train_model()
