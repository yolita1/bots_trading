# data_collector.py

# Import des bibliothèques nécessaires
import pandas as pd  # Bibliothèque pour la manipulation de données (notamment avec les DataFrames).
import asyncio  # Module pour gérer des tâches asynchrones.
import signal  # Module pour gérer les signaux système (comme l'arrêt du programme).
import sys  # Module système pour interagir avec le système d'exploitation.
from binance import AsyncClient, BinanceSocketManager  # Librairie Binance pour l'accès aux données et au trading en mode asynchrone.
from config import API_KEY, API_SECRET, SYMBOLS, INTERVAL  # Import des clés et paramètres depuis le fichier config.
from database import create_connection, create_tables, insert_market_data  # Import des fonctions pour interagir avec la base de données.
import logging
import sys
import asyncio
import aiohttp
from aiohttp import TCPConnector

# Utilisation de ProactorEventLoopPolicy pour Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Désactiver aiodns et utiliser le résolveur DNS par défaut de aiohttp
def create_session():
    return aiohttp.ClientSession(connector=TCPConnector(use_dns_cache=True))

# Modification de la méthode _init_client pour utiliser cette nouvelle session
async def _init_client(self):
    self.client = await AsyncClient.create(API_KEY, API_SECRET, session=create_session())  # Utiliser la session sans aiodns
    self.bm = BinanceSocketManager(self.client)

# Fonction pour configurer un logger pour enregistrer les événements
def setup_logger(name):
    # Configure le logger pour afficher les messages d'info et d'erreurs dans la console.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    return logger

# Initialisation du logger
logger = setup_logger(__name__)  # Crée un logger pour enregistrer les événements du fichier `data_collector.py`.

# Définition de la classe DataCollector pour gérer la collecte de données
class DataCollector:
    def __init__(self, symbols, interval):
        self.symbols = symbols  # Liste des symboles (paires de trading, par exemple BTCUSDT).
        self.interval = interval  # Intervalle de temps pour les données (par exemple '1m' pour une minute).
        self.client = None  # Initialement, le client Binance n'est pas encore créé.
        self.bm = None  # Le gestionnaire de socket n'est pas encore créé.
        self.streams = []  # Liste pour stocker les flux de données (streams).
        self.running = True  # Booléen pour contrôler si le collecteur de données est en cours d'exécution.

    # Méthode asynchrone pour initialiser le client Binance
    async def _init_client(self):
        self.client = await AsyncClient.create(API_KEY, API_SECRET)  # Crée un client Binance avec les clés API pour accéder aux données.
        self.bm = BinanceSocketManager(self.client)  # Initialise le gestionnaire de socket pour recevoir les flux de données de marché.

    # Méthode asynchrone pour fermer la connexion avec le client Binance
    async def _close_client(self):
        if self.client:
            await self.client.close_connection()  # Ferme la connexion avec le client Binance si elle est ouverte.

    # Méthode asynchrone pour collecter les données pour un symbole donné
    async def _collect_data(self, symbol):
        try:
            logger.info(f"Démarrage de la collecte pour {symbol}")  # Log l'information sur le démarrage de la collecte pour un symbole.
            ts = self.bm.kline_socket(symbol=symbol, interval=self.interval)  # Ouvre un flux (stream) de données de marché (kline) pour ce symbole.
            async with ts as tscm:
                while self.running:  # Tant que le collecteur est en cours d'exécution.
                    res = await tscm.recv()  # Reçoit des messages de données depuis le flux de données (kline).
                    if res:
                        kline = res['k']  # Récupère les données de la kline (bougie) qui contiennent les informations de trading.
                        if kline['x']:  # Vérifie si la bougie est clôturée (finalisée).
                            # Crée un dictionnaire contenant les informations de la bougie.
                            data = {
                                'symbol': kline['s'],  # Symbole du marché (par exemple, BTCUSDT).
                                'timestamp': pd.to_datetime(kline['t'], unit='ms'),  # Convertit l'horodatage en date.
                                'open': float(kline['o']),  # Prix d'ouverture de la bougie.
                                'high': float(kline['h']),  # Prix le plus élevé de la bougie.
                                'low': float(kline['l']),  # Prix le plus bas de la bougie.
                                'close': float(kline['c']),  # Prix de clôture de la bougie.
                                'volume': float(kline['v'])  # Volume échangé pendant cette période.
                            }
                            # Insère les données de marché dans la base de données.
                            insert_market_data([data])
                            logger.info(f"Nouvelle bougie pour {symbol} insérée dans la base de données.")  # Log l'insertion réussie.
        except Exception as e:
            # Log l'erreur si quelque chose se passe mal pendant la collecte de données pour ce symbole.
            logger.error(f"Erreur lors de la collecte des données pour {symbol} : {e}")

    # Méthode asynchrone pour collecter les données pour tous les symboles de la liste
    async def _collect_all_symbols(self):
        tasks = []  # Crée une liste de tâches asynchrones.
        for symbol in self.symbols:
            # Pour chaque symbole, crée une tâche asynchrone pour collecter les données.
            tasks.append(asyncio.create_task(self._collect_data(symbol)))
        await asyncio.gather(*tasks)  # Exécute toutes les tâches en parallèle.

    # Méthode pour gérer les signaux d'arrêt (SIGINT et SIGTERM)
    def _signal_handler(self, sig, frame):
        logger.info("Signal reçu, arrêt du collecteur de données...")  # Log lorsque le signal d'arrêt est reçu.
        self.running = False  # Arrête la collecte des données en mettant `self.running` à False.
        sys.exit(0)  # Arrête le programme proprement.

    # Méthode pour démarrer la collecte de données
    def start(self):
        # Enregistre les gestionnaires de signaux pour interrompre le programme proprement en cas de signal d'arrêt.
        signal.signal(signal.SIGINT, self._signal_handler)  # Interruption (Ctrl+C).
        signal.signal(signal.SIGTERM, self._signal_handler)  # Arrêt depuis le système.
        asyncio.run(self._run())  # Lance la boucle asynchrone pour démarrer le collecteur de données.

    # Méthode principale qui lance le client Binance, collecte les données et ferme le client
    async def _run(self):
        await self._init_client()  # Initialise le client Binance.
        await self._collect_all_symbols()  # Démarre la collecte de données pour tous les symboles.
        await self._close_client()  # Ferme la connexion avec le client Binance une fois terminé.

# Si ce fichier est exécuté directement (pas importé comme module)
if __name__ == "__main__":
    # Liste des symboles à collecter définie dans config.py (exemple : ['BTCUSDT', 'ETHUSDT']).
    symbols = SYMBOLS
    create_tables()  # Crée les tables nécessaires dans la base de données si elles n'existent pas encore.
    collector = DataCollector(symbols, INTERVAL)  # Crée une instance du collecteur de données avec les symboles et l'intervalle définis.
    collector.start()  # Démarre la collecte des données.
