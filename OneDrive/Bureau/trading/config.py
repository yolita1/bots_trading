# config.py

# Clés API pour accéder à Binance. Elles sont nécessaires pour que le bot puisse se connecter à ton compte et effectuer des transactions.
API_KEY = 'mY65BiR09AoDaOhlC2RHszlwOPPiZIuYnGHV6CdFq6cCb7YZPBArxDmX3iAKGEjr'  # Clé publique pour l'authentification API
API_SECRET = 'ehBhW6dmf41FJgTYoq6vfmfE600zkrdgMkgqoejB7c6tT9ukLJYOLm9CcTtKvaKz'  # Clé secrète pour l'authentification API

# Paramètres de trading
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Le symbole que tu veux trader. Ici, c'est le Bitcoin contre l'USDT (dollar Tether).
INTERVAL = '1m'  # L'intervalle de temps utilisé pour les données de marché. Ici, chaque bougie représente 1 minute.
LIMIT = 1000  # Le nombre maximum de données historiques récupérées (ici 1000 dernières bougies).

# Paramètres de position
QUANTITY = 0.001  # Quantité de BTC que tu veux acheter ou vendre par transaction (ici 0.001 BTC).
MAX_POSITION = 0.005  # Quantité maximale de BTC que tu peux posséder en position à tout moment (ici 0.005 BTC).

# Paramètres pour l'apprentissage automatique
MODEL_PATH = 'models/trading_model.h5'  # Chemin où est stocké le modèle d'apprentissage automatique utilisé pour prédire les trades.
TRAINING_DATA_PATH = 'data/training_data.csv'  # Chemin vers les données de formation utilisées pour entraîner le modèle.

# Paramètres de gestion des risques
MAX_DRAWDOWN = 0.1  # Le drawdown maximal autorisé avant d'arrêter le trading (ici 10% de perte maximale sur le capital total).
RISK_PER_TRADE = 0.02  # Pourcentage du capital risqué sur chaque trade (ici 2%).
MAX_RISK_PER_POSITION = 0.05  # Risque maximal pour chaque position ouverte en pourcentage du capital (ici 5%).

# Paramètres pour les notifications par e-mail
EMAIL = 'votre_email@example.com'  # L'adresse e-mail utilisée pour envoyer des notifications.
EMAIL_PASSWORD = 'votre_mot_de_passe_email'  # Le mot de passe associé à l'adresse e-mail.
SMTP_SERVER = 'smtp.example.com'  # Serveur SMTP utilisé pour envoyer les e-mails (à remplacer par le bon serveur de ton fournisseur).
SMTP_PORT = 587  # Port utilisé pour le serveur SMTP (587 est standard pour les connexions sécurisées).

# Chemin du fichier de log
LOG_FILE = 'logs/trading_bot.log'  # Chemin vers le fichier où seront enregistrées les actions et erreurs du bot (pour suivre ce qu'il fait).
