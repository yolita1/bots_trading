import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
import time
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connexion à l'API Binance
binance = ccxt.binance({
    'apiKey': os.getenv('mY65BiR09AoDaOhlC2RHszlwOPPiZIuYnGHV6CdFq6cCb7YZPBArxDmX3iAKGEjr'),
    'secret': os.getenv('ehBhW6dmf41FJgTYoq6vfmfE600zkrdgMkgqoejB7c6tT9ukLJYOLm9CcTtKvaKz'),
    'enableRateLimit': True,
})

# Convertir 10 euros en USDT
def get_investment_amount_in_usdt(euro_amount=10):
    euro_to_usdt = binance.fetch_ticker('EUR/USDT')['last']
    return euro_amount * euro_to_usdt

# Récupération des données de marché
def fetch_data(symbol, timeframe='1h', limit=1000):
    logging.info(f"Fetching market data for {symbol}")
    data = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Calcul des moyennes mobiles
def calculate_moving_averages(df, short_window=40, long_window=100):
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()
    return df

# Préparation des données pour LSTM
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

# Construction et entraînement du modèle LSTM
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=2)
    
    return model

# Prédiction avec le modèle LSTM
def predict_with_lstm(model, df, scaler):
    X, _, _ = prepare_data(df)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    last_prediction = predictions[-1]
    return last_prediction

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
    
    # Calcul des moyennes mobiles
    data = calculate_moving_averages(data)
    
    # Utilisation des moyennes mobiles pour filtrer les signaux LSTM
    if data['short_mavg'].iloc[-1] > data['long_mavg'].iloc[-1]:
        trend = 'up'
    else:
        trend = 'down'

    X_train, y_train, scaler = prepare_data(data)
    model = train_lstm_model(X_train, y_train)

    last_price_prediction = predict_with_lstm(model, data, scaler)
    current_price = data['close'].iloc[-1]

    if last_price_prediction > current_price and trend == 'up':
        decision = 'buy'
        stop_loss_price = current_price * 0.98  # Stop-loss à 2% en dessous du prix actuel
        take_profit_price = current_price * 1.05  # Take-profit à 5% au-dessus du prix actuel
    elif last_price_prediction < current_price and trend == 'down':
        decision = 'sell'
        stop_loss_price = current_price * 1.02  # Stop-loss à 2% au-dessus du prix actuel
        take_profit_price = current_price * 0.95  # Take-profit à 5% en dessous du prix actuel
    else:
        decision = None

    if decision:
        investment_amount = get_investment_amount_in_usdt(10) / current_price  # Utiliser 10 euros convertis en USDT
        execute_order(symbol, decision, amount=investment_amount, stop_loss=stop_loss_price, take_profit=take_profit_price)

if __name__ == "__main__":
    main()
