import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, LayerNormalization
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize, differential_evolution
import logging
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import requests
import prometheus_client as prom
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration des métriques Prometheus
trade_counter = prom.Counter('trades_executed', 'Number of trades executed')
profit_gauge = prom.Gauge('profit', 'Profit from trades')
loss_gauge = prom.Gauge('loss', 'Loss from trades')
error_counter = prom.Counter('errors', 'Number of errors encountered')

# Connexion à l'API Binance avec une option de Testnet pour la sécurité
use_testnet = True
binance = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
})
if use_testnet:
    binance.set_sandbox_mode(True)  # Active le mode testnet pour sécuriser les tests

# Convertir 10 euros en USDT
def get_investment_amount_in_usdt(euro_amount=10):
    euro_to_usdt = binance.fetch_ticker('EUR/USDT')['last']
    return euro_amount * euro_to_usdt

# Récupération des données de marché
def fetch_data(symbol, timeframe='1h', limit=1000):
    logging.info(f"Fetching market data for {symbol}")
    try:
        data = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {e}")
        error_counter.inc()
        return pd.DataFrame()

# Préparation des données pour LSTM
def prepare_data(df, sequence_length=60):
    logging.info("Preparing data for LSTM model")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Construction et entraînement du modèle LSTM avancé avec Attention
def train_lstm_model(X_train, y_train):
    logging.info("Training advanced LSTM model with Attention")
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Attention())
    model.add(LayerNormalization())
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=2)
    
    return model

# Modèle d'Ensemble : RandomForest + Gradient Boosting
def train_ensemble_models(X_train, y_train):
    logging.info("Training ensemble models (Random Forest + Gradient Boosting)")
    rf_model = RandomForestClassifier(n_estimators=100)
    gb_model = GradientBoostingClassifier(n_estimators=100)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model

# Prédiction avec le modèle LSTM
def predict_with_lstm(model, df, scaler):
    logging.info("Making predictions with LSTM model")
    X, _, _ = prepare_data(df)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    last_prediction = predictions[-1]
    return last_prediction

# Prédiction avec les modèles d'ensemble
def predict_with_ensemble(models, X_test):
    logging.info("Making predictions with ensemble models")
    rf_model, gb_model = models
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    # Combiner les prédictions des deux modèles
    ensemble_pred = (rf_pred + gb_pred) / 2
    return ensemble_pred

# Analyse de sentiment via NLP avec Transformers
def sentiment_analysis(news_headlines):
    logging.info("Performing sentiment analysis with NLP")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    inputs = tokenizer(news_headlines, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    
    sentiment_scores = predictions[:, 1].numpy()
    avg_sentiment = np.mean(sentiment_scores)
    
    return avg_sentiment

# Optimisation du portefeuille avec des méthodes de métaheuristique
def optimize_portfolio(mean_returns, cov_matrix):
    logging.info("Optimizing portfolio using metaheuristic method (Differential Evolution)")
    
    def sharpe_ratio(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - 0.01) / port_volatility  # Minimiser l'opposé du Sharpe ratio

    num_assets = len(mean_returns)
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    result = differential_evolution(sharpe_ratio, bounds, constraints=constraints)
    
    return result.x

# Exécution d'ordres avec des stratégies d'exécution intelligente
def execute_order(symbol, side, amount, strategy='vwap'):
    try:
        logging.info(f"Executing order: {side} {amount} of {symbol} using {strategy} strategy")
        if strategy == 'vwap':
            data = fetch_data(symbol, '1m', limit=500)
            total_volume = data['volume'].sum()
            target_vwap = np.sum(data['close'] * data['volume']) / total_volume
            remaining_volume = amount
            for _, row in data.iterrows():
                if remaining_volume <= 0:
                    break
                trade_volume = min(remaining_volume, row['volume'])
                binance.create_market_order(symbol, side, trade_volume)
                remaining_volume -= trade_volume
                time.sleep(1)
        elif strategy == 'ladder':
            for i in range(5):
                partial_amount = amount / 5
                binance.create_market_order(symbol, side, partial_amount)
                time.sleep(1)
        else:
            binance.create_market_order(symbol, side, amount)
        logging.info(f"Order executed: {side} {amount} of {symbol}")
        trade_counter.inc()
    except Exception as e:
        logging.error(f"Order execution failed: {e}")
        error_counter.inc()

# Environnement pour l'apprentissage par renforcement (Reinforcement Learning)
def create_rl_env():
    logging.info("Creating Reinforcement Learning Environment")
    env = make_vec_env(lambda: TradingEnv(binance, symbol='BTC/USDT', initial_balance=10000), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1)
    return model

# Exécuter des stratégies basées sur l'apprentissage par renforcement
def execute_rl_strategy(model, steps=10000):
    logging.info(f"Executing Reinforcement Learning strategy for {steps} steps")
    model.learn(total_timesteps=steps)
    model.save("rl_trading_model")
    return model

# Surveillance des performances et alertes
def monitor_performance():
    logging.info("Monitoring bot performance...")
    # Placeholder pour la surveillance des performances du bot. 
    # Ce code pourrait être étendu pour inclure des alertes et des rapports sur les performances en temps réel.

# Environnement parallèle pour exécution multiple
def parallel_execution(func, *args, **kwargs):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(partial(func, *args, **kwargs))
        return future.result()



# Fonction principale du bot
def main():
    symbol = 'BTC/USDT'
    data = fetch_data(symbol)
    X_train, y_train, scaler = prepare_data(data)
    
    # Entraîner les modèles d'ensemble
    ensemble_models = parallel_execution(train_ensemble_models, X_train, y_train)
    
    # Entraîner le modèle LSTM
    lstm_model = parallel_execution(train_lstm_model, X_train, y_train)

    news_headlines = [
        "Bitcoin hits new highs amid market optimism",
        "Regulation fears could trigger another crypto crash",
        "Institutional investors pour money into Bitcoin"
    ]
    sentiment_score = parallel_execution(sentiment_analysis, news_headlines)

    # Prédiction avec les modèles d'ensemble
    ensemble_predictions = parallel_execution(predict_with_ensemble, ensemble_models, X_train)
    
    # Prédiction avec le modèle LSTM
    last_price_prediction = parallel_execution(predict_with_lstm, lstm_model, data, scaler)
    
    current_price = data['close'].iloc[-1]

    decision = None
    if sentiment_score > 0.6 and last_price_prediction > current_price:
        decision = 'buy'
        send_telegram_alert(f"Buy signal detected for {symbol} at price {current_price}")
    elif sentiment_score < 0.4 and last_price_prediction < current_price:
        decision = 'sell'
        send_telegram_alert(f"Sell signal detected for {symbol} at price {current_price}")

    if decision:
        investment_amount = get_investment_amount_in_usdt()  # Utilise 10 euros convertis en USDT
        parallel_execution(execute_order, symbol, decision, amount=investment_amount, strategy='vwap')

    # Exécuter des stratégies basées sur l'apprentissage par renforcement
    rl_model = create_rl_env()
    parallel_execution(execute_rl_strategy, rl_model, steps=10000)

    assets = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    returns_data = {asset: fetch_data(asset)['close'].pct_change().dropna() for asset in assets}
    returns_df = pd.DataFrame(returns_data)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    optimal_weights = parallel_execution(optimize_portfolio, mean_returns, cov_matrix)
    logging.info(f"Optimal portfolio weights: {optimal_weights}")

    monitor_performance()

if __name__ == "__main__":
    prom.start_http_server(8000)
    while True:
        main()
        time.sleep(3600)  # Exécuter toutes les heures
