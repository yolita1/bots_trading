# Importation des bibliothèques nécessaires
import pandas as pd  # pandas est une bibliothèque pour manipuler des données sous forme de tableaux.
import numpy as np  # numpy est utilisé pour effectuer des calculs mathématiques sur des matrices ou tableaux.
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler est un outil pour normaliser les données entre 0 et 1.
from logger import setup_logger  # Importation de la fonction pour configurer le système de suivi des événements (logger).

# Création d'un logger pour enregistrer les événements (informations, erreurs) pendant l'exécution du code.
logger = setup_logger(__name__)

# Définition de la classe DataPreprocessor pour prétraiter les données.
class DataPreprocessor:
    # Initialisation de l'objet DataPreprocessor. Cette méthode est appelée automatiquement à la création de l'objet.
    def __init__(self):
        self.scaler = None  # Création d'un attribut `scaler` pour normaliser les données plus tard.

    # Méthode pour prétraiter les données passées en argument.
    def preprocess_data(self, data):
        try:
            data = data.copy()  # Copie les données pour ne pas modifier l'original.
            data = self._calculate_indicators(data)  # Calcul des indicateurs techniques (RSI, MACD, etc.).
            data = data.dropna()  # Supprime les lignes contenant des valeurs manquantes (NaN).
            
            # Sélection des colonnes d'intérêt pour les prédictions (features).
            features = data[['close', 'MACD', 'Signal', 'RSI', 'Stochastic_K', 'Stochastic_D', 'ADX',
                             'Bollinger_Upper', 'Bollinger_Lower', 'Momentum', 'WilliamsR']]
                             
            # Initialisation du scaler pour normaliser les valeurs entre 0 et 1.
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Applique la normalisation sur les colonnes sélectionnées (features).
            scaled_features = self.scaler.fit_transform(features)
            
            logger.info("Données prétraitées avec succès.")  # Enregistre une info dans le logger pour indiquer que tout s'est bien passé.
            return scaled_features  # Retourne les données normalisées.
        except Exception as e:  # Si une erreur se produit à l'intérieur du try, elle est capturée ici.
            logger.error(f"Erreur lors du prétraitement des données : {e}")  # Enregistre l'erreur dans le logger.
            return None  # Retourne None en cas d'erreur.

    # Méthode privée pour calculer différents indicateurs techniques sur les données (comme MACD, RSI, etc.).
    def _calculate_indicators(self, data):
        data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()  # Calcul de l'EMA (Exponential Moving Average) sur 12 périodes.
        data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()  # Calcul de l'EMA sur 26 périodes.
        
        # Calcul de la différence entre EMA12 et EMA26, qui est le MACD.
        data['MACD'] = data['EMA12'] - data['EMA26']
        
        # Calcul de la ligne de signal, qui est une EMA du MACD sur 9 périodes.
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        data['RSI'] = self._compute_RSI(data['close'])  # Calcul de l'indicateur RSI (Relative Strength Index).
        
        # Calcul des deux lignes %K et %D de l'oscillateur stochastique.
        data['Stochastic_K'], data['Stochastic_D'] = self._compute_stochastic_oscillator(data)
        
        # Calcul de l'indicateur ADX (Average Directional Index).
        data['ADX'] = self._compute_ADX(data)
        
        # Calcul des bandes de Bollinger (moyenne, bande supérieure, bande inférieure).
        data['Bollinger_Middle'], data['Bollinger_Upper'], data['Bollinger_Lower'] = self._compute_bollinger_bands(data)
        
        # Calcul du Momentum, qui mesure la vitesse de changement du prix.
        data['Momentum'] = data['close'] - data['close'].shift(10)
        
        # Calcul de l'indicateur Williams %R, qui mesure les conditions de surachat ou survente.
        data['WilliamsR'] = self._compute_williams_r(data)

        return data  # Retourne les données enrichies avec les indicateurs techniques.

    # Méthode pour calculer le RSI (Relative Strength Index).
    def _compute_RSI(self, series, period=14):
        delta = series.diff()  # Calcul de la différence entre les prix de clôture consécutifs.
        gain = delta.where(delta > 0, 0.0)  # Garde les gains (variations positives).
        loss = -delta.where(delta < 0, 0.0)  # Garde les pertes (variations négatives).
        
        # Moyenne exponentielle des gains et des pertes.
        avg_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
        avg_loss = loss.ewm(com=(period - 1), min_periods=period).mean()
        
        # Calcul du rapport gains/pertes.
        rs = avg_gain / avg_loss
        
        # Calcul du RSI basé sur le rapport gains/pertes.
        rsi = 100 - (100 / (1 + rs))
        return rsi  # Retourne le RSI.

    # Méthode pour calculer l'oscillateur stochastique (%K et %D).
    def _compute_stochastic_oscillator(self, data, period=14):
        lowest_low = data['low'].rolling(window=period).min()  # Trouve le plus bas sur `period` périodes.
        highest_high = data['high'].rolling(window=period).max()  # Trouve le plus haut sur `period` périodes.
        
        # Calcul de %K (rapport entre la différence close-min et high-low).
        stochastic_K = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        
        # %D est la moyenne mobile de %K sur 3 périodes.
        stochastic_D = stochastic_K.rolling(window=3).mean()
        return stochastic_K, stochastic_D  # Retourne les deux lignes stochastiques %K et %D.

    # Méthode pour calculer l'ADX (Average Directional Index).
    def _compute_ADX(self, data, period=14):
        plus_dm = data['high'].diff()  # Calcul des directions positives (variations positives des plus hauts).
        minus_dm = -data['low'].diff()  # Calcul des directions négatives (variations négatives des plus bas).
        
        # Remplacer les valeurs négatives par 0.
        plus_dm[plus_dm < 0] = 0.0
        minus_dm[minus_dm < 0] = 0.0

        # Calcul du True Range (plage de variation du prix).
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calcul de l'ATR (Average True Range) sur `period` périodes.
        atr = true_range.rolling(window=period).mean()
        
        # Calcul des indicateurs DI+ et DI- (directions positives et négatives).
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calcul de DX (Difference Index).
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        
        # Calcul de l'ADX (moyenne mobile de DX).
        adx = dx.rolling(window=period).mean()
        return adx  # Retourne l'ADX.

    # Méthode pour calculer les bandes de Bollinger.
    def _compute_bollinger_bands(self, data, period=20, std_dev=2):
        middle_band = data['close'].rolling(window=period).mean()  # Calcul de la moyenne mobile centrale.
        std = data['close'].rolling(window=period).std()  # Calcul de l'écart type des prix.
        
        # Calcul des bandes supérieure et inférieure.
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        return middle_band, upper_band, lower_band  # Retourne les trois bandes de Bollinger.

    # Méthode pour calculer l'indicateur Williams %R.
    def _compute_williams_r(self, data, period=14):
        highest_high = data['high'].rolling(window=period).max()  # Le plus haut sur `period` périodes.
        lowest_low = data['low'].rolling(window=period).min()  # Le plus bas sur `period` périodes.
        
        # Calcul du Williams %R.
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        return williams_r  # Retourne le Williams %R.

    # Méthode pour inverser la normalisation (ramener les données normalisées à leur échelle d'origine).
    def inverse_transform(self, scaled_data):
        return self.scaler.inverse_transform(scaled_data)  # Applique l'inversion du scaler sur les données normalisées.
