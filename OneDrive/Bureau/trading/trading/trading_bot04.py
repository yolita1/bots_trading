import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
import logging
import os
import requests

# Assurez-vous que les clés API sont stockées en tant que variables d'environnement
api_key = os.getenv('mY65BiR09AoDaOhlC2RHszlwOPPiZIuYnGHV6CdFq6cCb7YZPBArxDmX3iAKGEjr')
api_secret = os.getenv('ehBhW6dmf41FJgTYoq6vfmfE600zkrdgMkgqoejB7c6tT9ukLJYOLm9CcTtKvaKz')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connexion à l'API Binance
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# Test de la connexion à l'API en récupérant le ticker BTC/USDT
try:
    ticker = binance.fetch_ticker('BTC/USDT')
    print(f"Connexion réussie, prix actuel du BTC/USDT : {ticker['last']}")
except Exception as e:
    print(f"Erreur de connexion à l'API Binance : {e}")

# Convertir 10 USD en USDT
def get_investment_amount_in_usdt(dollar_amount=10):
    return dollar_amount  # En supposant que vous voulez utiliser exactement 10 USDT

# Récupération des données de marché
def fetch_data(symbol, timeframe='1h', limit=1000):
    logging.info(f"Fetching market data for {symbol}")
    data = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Préparation des données pour les modèles
def prepare_data(df, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Construction du modèle LSTM avancé
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Exécution d'ordre avec gestion du risque
def execute_order(symbol, side, amount, stop_loss=None, take_profit=None):
    try:
        logging.info(f"Executing order: {side} {amount} of {symbol}")
        order = binance.create_market_order(symbol, side, amount)
        
        if stop_loss:
            binance.create_order(symbol, 'STOP_LOSS', side, amount, stop_loss)
            logging.info(f"Stop-loss set at {stop_loss}")
        
        if take_profit:
            binance.create_order(symbol, 'TAKE_PROFIT', side, amount, take_profit)
            logging.info(f"Take-profit set at {take_profit}")
            
        return order
    except Exception as e:
        logging.error(f"Order execution failed: {e}")

# Fonction principale du bot
def main():
    symbol = 'BTC/USDT'
    data = fetch_data(symbol)
    
    X, y, scaler = prepare_data(data)
    
    # Entraîner les modèles d'ensemble
    lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=15, batch_size=64, verbose=2)

    last_price_prediction = lstm_model.predict(X)
    last_price_prediction = scaler.inverse_transform(last_price_prediction)
    current_price = data['close'].iloc[-1]

    if last_price_prediction[-1] > current_price:
        decision = 'buy'
        stop_loss_price = current_price * 0.98  # Stop-loss à 2% en dessous du prix actuel
        take_profit_price = current_price * 1.05  # Take-profit à 5% au-dessus du prix actuel
    elif last_price_prediction[-1] < current_price:
        decision = 'sell'
        stop_loss_price = current_price * 1.02  # Stop-loss à 2% au-dessus du prix actuel
        take_profit_price = current_price * 0.95  # Take-profit à 5% en dessous du prix actuel
    else:
        decision = None

    if decision:
        investment_amount = get_investment_amount_in_usdt(10) / current_price  # Utiliser 10 USD convertis en USDT
        execute_order(symbol, decision, amount=investment_amount, stop_loss=stop_loss_price, take_profit=take_profit_price)

if __name__ == "__main__":
    main()