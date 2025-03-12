# Importation des bibliothèques nécessaires
import asyncio  # Utilisé pour exécuter des tâches asynchrones (permet au bot de traiter plusieurs symboles en parallèle).
import signal  # Utilisé pour capturer les signaux du système (comme l'arrêt du programme).
import sys  # Utilisé pour interagir avec le système (comme pour arrêter le programme).
import pandas as pd  # Utilisé pour manipuler des tableaux de données (comme les données de marché).
from config import SYMBOLS, MAX_DRAWDOWN, DATA_PATH  # Import des paramètres de configuration.
from data_collector import DataCollector  # Import de la classe pour collecter les données de marché.
from strategy import generate_signal  # Import de la fonction pour générer un signal de trading (BUY/SELL/HOLD).
from trader import Trader  # Import de la classe pour exécuter les ordres de trading.
from notifier import Notifier  # Import de la classe pour envoyer des notifications.
from risk_manager import RiskManager  # Import de la classe pour gérer les risques.
from logger import setup_logger  # Importation du système de suivi des événements (logger).

# Ajout d'une variable globale pour stocker les informations de trading en temps réel
trading_data = {}

# Configuration du logger pour enregistrer les événements et erreurs
logger = setup_logger(__name__)

# Définition de la classe TradingBot, qui gère l'ensemble du bot de trading
class TradingBot:
    def __init__(self):
        self.symbols = SYMBOLS  # Liste des symboles que le bot va trader (par exemple, BTCUSDT, ETHUSDT).
        self.data_collectors = []  # Liste pour stocker les collecteurs de données pour chaque symbole.
        self.traders = {}  # Dictionnaire pour stocker les objets Trader pour chaque symbole.
        self.risk_managers = {}  # Dictionnaire pour stocker les objets RiskManager pour chaque symbole.
        self.notifier = Notifier()  # Instanciation de l'objet Notifier pour envoyer des notifications.
        self.running = True  # Booléen pour contrôler si le bot est en cours d'exécution.

    # Méthode asynchrone pour démarrer les collecteurs de données
    async def start_data_collectors(self):
        for symbol in self.symbols:
            # Crée un collecteur de données pour chaque symbole avec un intervalle d'une minute.
            data_collector = DataCollector(symbols=[symbol], interval='1m')
            self.data_collectors.append(data_collector)  # Ajoute le collecteur à la liste.
            asyncio.create_task(data_collector._run())  # Démarre la collecte de données en arrière-plan.

    # Méthode principale pour lancer le bot
    async def run(self):
        await self.start_data_collectors()  # Démarre les collecteurs de données.
        await asyncio.sleep(5)  # Attendre 5 secondes pour que les collecteurs aient suffisamment de données.

        while self.running:
            tasks = []  # Liste pour stocker les tâches asynchrones pour chaque symbole.
            for symbol in self.symbols:
                tasks.append(self.process_symbol(symbol))  # Ajoute une tâche pour chaque symbole.
            await asyncio.gather(*tasks)  # Exécute toutes les tâches en parallèle.
            await asyncio.sleep(60)  # Attendre 60 secondes avant de traiter à nouveau les symboles.

    # Méthode pour traiter les données et exécuter des trades pour un symbole donné
    async def process_symbol(self, symbol):
        try:
            # Charge les données de marché pour le symbole
            data = self.load_data(symbol)
            if data is None:  # Si les données sont manquantes, retourner (ne rien faire).
                return

            # Si le RiskManager ou le Trader pour ce symbole n'est pas encore créé, on les initialise
            if symbol not in self.risk_managers:
                self.risk_managers[symbol] = RiskManager(symbol)  # Crée un RiskManager pour gérer les risques.
            if symbol not in self.traders:
                self.traders[symbol] = Trader(symbol)  # Crée un Trader pour exécuter les ordres.

            risk_manager = self.risk_managers[symbol]
            trader = self.traders[symbol]

            # Vérifie si le drawdown maximal n'est pas dépassé avant de trader
            if not risk_manager.check_drawdown():
                logger.warning(f"Drawdown maximal atteint pour {symbol}. Trading suspendu.")  # Log si le drawdown est trop élevé.
                return

            # Générer le signal de trading (BUY, SELL, HOLD)
            signal = generate_signal(data, symbol)

            # Calculer la taille de la position en fonction du risque
            stop_loss_percentage = 0.02  # Définir un stop-loss de 2% pour cet exemple.
            position_size = risk_manager.perform_risk_management(stop_loss_percentage)

            if position_size > 0:
                # Exécuter le trade si la taille de position est suffisante
                await trader.execute_trade(signal, position_size)

                # Mettre à jour les données de trading après le trade
                trading_data[symbol] = {
                    'signal': signal,
                    'position_size': position_size,
                    'timestamp': datetime.utcnow().isoformat()  # Mettre à jour avec l'heure actuelle
                }

                # Envoyer une notification avec les détails du trade
                message = f"Signal: {signal} pour {symbol} avec une taille de position de {position_size:.4f}."
                await self.notifier.notify(f"Signal de Trading pour {symbol}", message)
            else:
                logger.info(f"Aucun trade placé pour {symbol} en raison des restrictions de gestion des risques.")  # Log si aucun trade n'est placé.

        except Exception as e:
            logger.error(f"Erreur lors du traitement du symbole {symbol}: {e}")  # Log toute erreur inattendue lors du traitement.

    # Méthode pour charger les données de marché depuis un fichier CSV
    def load_data(self, symbol):
        try:
            # Charge les données pour le symbole à partir d'un fichier CSV.
            file_path = f"{DATA_PATH}/{symbol}_live_data.csv"
            data = pd.read_csv(file_path)
            return data  # Retourne les données.
        except FileNotFoundError:  # Si le fichier n'existe pas, log un avertissement.
            logger.warning(f"Données introuvables pour {symbol}.")
            return None
        except Exception as e:  # Capture et log toute autre erreur.
            logger.error(f"Erreur lors du chargement des données pour {symbol}: {e}")
            return None

    # Méthode pour arrêter le bot de trading proprement
    def stop(self):
        self.running = False  # Met le bot en état d'arrêt.
        for collector in self.data_collectors:
            collector.running = False  # Arrête tous les collecteurs de données.
        logger.info("Arrêt du bot de trading.")  # Log l'arrêt du bot.

# Fonction principale pour démarrer le bot
def main():
    bot = TradingBot()  # Instanciation du bot de trading.

    # Gestion des signaux système pour arrêter proprement le bot
    def signal_handler(sig, frame):
        logger.info("Signal reçu, arrêt du bot...")  # Log la réception du signal d'arrêt.
        bot.stop()  # Arrête le bot.
        sys.exit(0)  # Ferme le programme.

    # Capture des signaux d'interruption (Ctrl+C) et de terminaison
    signal.signal(signal.SIGINT, signal_handler)  # Capture Ctrl+C pour arrêter le bot.
    signal.signal(signal.SIGTERM, signal_handler)  # Capture la demande d'arrêt depuis le système.

    asyncio.run(bot.run())  # Lance le bot en mode asynchrone.

# Point d'entrée du programme
if __name__ == "__main__":
    main()  # Appelle la fonction principale pour démarrer le bot.
