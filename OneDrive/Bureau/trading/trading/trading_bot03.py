import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import logging
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connexion à l'API Binance
binance = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
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

# Construction du modèle CNN pour l'ensemble
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Entraînement et validation croisée
def train_ensemble_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    cnn_model = build_cnn_model((X_train.shape[1], X_train.shape[2]))
    
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=2)
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=2)
    
    lstm_pred = lstm_model.predict(X_test)
    cnn_pred = cnn_model.predict(X_test)
    
    ensemble_pred = (lstm_pred + cnn_pred) / 2
    ensemble_error = mean_squared_error(y_test, ensemble_pred)
    
    logging.info(f'Ensemble model error: {ensemble_error}')
    
    return lstm_model, cnn_model

# Prédiction avec les modèles LSTM et CNN
def predict_ensemble(lstm_model, cnn_model, X, scaler):
    lstm_pred = lstm_model.predict(X)
    cnn_pred = cnn_model.predict(X)
    
    ensemble_pred = (lstm_pred + cnn_pred) / 2
    predictions = scaler.inverse_transform(ensemble_pred)
    return predictions[-1]

# Environnement d'apprentissage par renforcement
def create_rl_env(symbol, initial_balance=10000):
    env = make_vec_env(lambda: TradingEnv(binance, symbol=symbol, initial_balance=initial_balance), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1)
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
    lstm_model, cnn_model = train_ensemble_models(X, y)

    last_price_prediction = predict_ensemble(lstm_model, cnn_model, X, scaler)
    current_price = data['close'].iloc[-1]

    if last_price_prediction > current_price:
        decision = 'buy'
        stop_loss_price = current_price * 0.98  # Stop-loss à 2% en dessous du prix actuel
        take_profit_price = current_price * 1.05  # Take-profit à 5% au-dessus du prix actuel
    elif last_price_prediction < current_price:
        decision = 'sell'
        stop_loss_price = current_price * 1.02  # Stop-loss à 2% au-dessus du prix actuel
        take_profit_price = current_price * 0.95  # Take-profit à 5% en dessous du prix actuel
    else:
        decision = None

    if decision:
        investment_amount = get_investment_amount_in_usdt(10) / current_price  # Utiliser 10 euros convertis en USDT
        execute_order(symbol, decision, amount=investment_amount, stop_loss=stop_loss_price, take_profit=take_profit_price)

    # Apprentissage par renforcement pour optimiser la stratégie
    rl_model = create_rl_env(symbol)
    rl_model.learn(total_timesteps=10000)
    rl_model.save("rl_trading_model")

if __name__ == "__main__":
    main()
