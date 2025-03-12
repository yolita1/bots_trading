# Importation des bibliothèques nécessaires
import math  # Utilisé pour les calculs mathématiques comme les arrondis et les logarithmes.
import time  # Utilisé pour mettre en pause l'exécution du programme (par exemple, lors des erreurs de l'API).
from binance.client import Client  # Client officiel de Binance pour interagir avec l'API.
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceRequestException, BinanceWithdrawException  # Gestion des différentes exceptions de l'API Binance.
from config import API_KEY, API_SECRET, SYMBOL  # Importation des clés API et du symbole de trading depuis le fichier config.
from logger import setup_logger  # Importation du système de suivi des événements (logger) pour enregistrer les erreurs et les informations.

# Configuration du logger pour enregistrer les événements importants.
logger = setup_logger(__name__)

# Classe Trader pour interagir avec l'API Binance et exécuter des ordres de trading.
class Trader:
    def __init__(self):
        self.client = self._init_client()  # Initialise la connexion à l'API Binance.
        self.symbol = SYMBOL  # Définit le symbole de trading (par exemple, 'BTCUSDT').
        self.asset = self.symbol.replace('USDT', '')  # Extrait l'actif principal du symbole (par exemple, 'BTC' de 'BTCUSDT').
        self.precision = self._get_symbol_precision()  # Détermine la précision de prix du symbole.
        self.step_size = self._get_symbol_step_size()  # Obtient la taille minimale des lots pour ce symbole.
        self.min_notional = self._get_min_notional()  # Obtient le montant minimum pour passer un ordre.

    # Méthode pour initialiser le client Binance en utilisant les clés API.
    def _init_client(self):
        try:
            client = Client(API_KEY, API_SECRET)  # Création du client Binance avec les clés API.
            logger.info("Connexion réussie à l'API Binance.")  # Log l'information si la connexion est réussie.
            return client
        except BinanceAPIException as e:  # Capture des erreurs spécifiques à l'API Binance.
            logger.error(f"Erreur lors de la connexion à l'API Binance : {e}")  # Log une erreur si la connexion échoue.
            time.sleep(5)  # Met en pause l'exécution pendant 5 secondes avant de réessayer.
            return self._init_client()  # Retente la connexion après la pause.

    # Méthode pour récupérer la précision de prix du symbole (le nombre de décimales autorisées).
    def _get_symbol_precision(self):
        try:
            info = self.client.get_symbol_info(self.symbol)  # Récupère les informations du symbole depuis l'API.
            for filt in info['filters']:
                if filt['filterType'] == 'PRICE_FILTER':  # Cherche le filtre qui concerne la précision de prix.
                    tick_size = float(filt['tickSize'])  # Récupère la taille minimale d'un tick de prix.
                    precision = int(round(-math.log(tick_size, 10), 0))  # Calcule le nombre de décimales.
                    logger.info(f"Précision du symbole {self.symbol} : {precision}")  # Log la précision calculée.
                    return precision
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération de la précision du symbole : {e}")  # Log une erreur en cas d'échec.
        return 6  # Retourne une valeur par défaut si la récupération échoue.

    # Méthode pour récupérer la taille minimale des lots (step size).
    def _get_symbol_step_size(self):
        try:
            info = self.client.get_symbol_info(self.symbol)  # Récupère les informations du symbole.
            for filt in info['filters']:
                if filt['filterType'] == 'LOT_SIZE':  # Cherche le filtre qui concerne la taille des lots.
                    step_size = float(filt['stepSize'])  # Récupère la taille minimale d'un lot.
                    logger.info(f"Step size du symbole {self.symbol} : {step_size}")  # Log la taille du lot.
                    return step_size
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération du step size du symbole : {e}")  # Log une erreur en cas d'échec.
        return 0.000001  # Valeur par défaut en cas d'erreur.

    # Méthode pour récupérer le montant minimal (notional) nécessaire pour passer un ordre.
    def _get_min_notional(self):
        try:
            info = self.client.get_symbol_info(self.symbol)  # Récupère les informations du symbole.
            for filt in info['filters']:
                if filt['filterType'] == 'MIN_NOTIONAL':  # Cherche le filtre concernant le montant minimal.
                    min_notional = float(filt['minNotional'])  # Récupère le montant minimal pour passer un ordre.
                    logger.info(f"Min notional pour {self.symbol} : {min_notional}")  # Log le montant minimal.
                    return min_notional
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération du min notional : {e}")  # Log une erreur en cas d'échec.
        return 10  # Valeur par défaut en cas d'erreur.

    # Méthode pour obtenir le solde total en USDT sur le compte.
    def get_account_balance(self):
        try:
            balance = self.client.get_asset_balance(asset='USDT')  # Récupère le solde en USDT.
            total_balance = float(balance['free']) + float(balance['locked'])  # Total = solde libre + solde bloqué.
            logger.info(f"Solde total du compte : {total_balance:.2f} USDT")  # Log le solde total du compte.
            return total_balance  # Retourne le solde total.
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération du solde du compte : {e}")  # Log une erreur en cas d'échec.
            return 0.0  # Retourne 0 en cas d'erreur.

    # Méthode pour obtenir le solde de l'actif principal (par exemple, BTC).
    def get_asset_balance(self):
        try:
            balance = self.client.get_asset_balance(asset=self.asset)  # Récupère le solde de l'actif principal.
            total_balance = float(balance['free']) + float(balance['locked'])  # Total = solde libre + bloqué.
            logger.info(f"Solde total pour {self.asset} : {total_balance}")  # Log le solde total de l'actif.
            return total_balance  # Retourne le solde total.
        except BinanceAPIException as e:
            logger.error(f"Erreur lors de la récupération du solde pour {self.asset} : {e}")  # Log une erreur en cas d'échec.
            return 0.0  # Retourne 0 en cas d'erreur.

    # Méthode pour calculer la quantité d'actif à acheter en fonction d'un montant en USDT.
    def calculate_quantity(self, usdt_amount):
        price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])  # Récupère le prix actuel du symbole.
        quantity = usdt_amount / price  # Calcule la quantité d'actif à acheter avec `usdt_amount`.
        # Ajuste la quantité pour qu'elle corresponde au step size minimum.
        precision = int(round(-math.log(self.step_size, 10), 0))  # Calcule le nombre de décimales.
        quantity = math.floor(quantity / self.step_size) * self.step_size  # Arrondit la quantité au step size le plus proche.
        quantity = float(round(quantity, precision))  # Arrondit la quantité à la précision correcte.
        logger.info(f"Quantité calculée : {quantity}")  # Log la quantité calculée.
        return quantity  # Retourne la quantité d'actif à acheter.

    # Méthode pour placer un ordre d'achat ou de vente sur Binance.
    def place_order(self, side, quantity, stop_loss=None, take_profit=None):
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,  # Le côté de l'ordre (achat ou vente).
                type='MARKET',  # Type d'ordre : ici, c'est un ordre de marché (achat/vente immédiat).
                quantity=quantity  # La quantité d'actif à acheter ou vendre.
            )
            logger.info(f"Ordre {side} exécuté : {order}")  # Log l'ordre exécuté.

            # Si un stop-loss et un take-profit sont fournis, placer un ordre OCO (ordre conditionnel).
            if stop_loss and take_profit:
                price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])  # Récupère le prix actuel du marché.
                # Calcule le prix de stop et de limite en fonction du stop-loss.
                stop_price = round(price * (1 - stop_loss) if side == 'BUY' else price * (1 + stop_loss), self.precision)
                limit_price = round(stop_price * (1 - 0.001) if side == 'BUY' else stop_price * (1 + 0.001), self.precision)
                # Calcule le prix de take-profit.
                take_profit_price = round(price * (1 + take_profit) if side == 'BUY' else price * (1 - take_profit), self.precision)

                # Crée un ordre OCO (ordre à déclenchement multiple).
                oco_order = self.client.create_oco_order(
                    symbol=self.symbol,
                    side='SELL' if side == 'BUY' else 'BUY',  # Place un ordre inverse (si achat, on vend; si vente, on achète).
                    quantity=quantity,  # Quantité à vendre ou acheter.
                    price=take_profit_price,  # Prix pour le take-profit.
                    stopPrice=stop_price,  # Prix pour déclencher le stop-loss.
                    stopLimitPrice=limit_price,  # Prix limite pour le stop-loss.
                    stopLimitTimeInForce='GTC'  # GTC signifie "Good 'Til Canceled" (ordre valable jusqu'à annulation).
                )
                logger.info(f"Ordre OCO placé : {oco_order}")  # Log l'ordre OCO placé.

            return order  # Retourne l'ordre exécuté.
        except BinanceAPIException as e:  # Capture des erreurs liées à l'API Binance.
            logger.error(f"Erreur lors de l'exécution de l'ordre {side}: {e}")  # Log une erreur en cas d'échec.
            time.sleep(5)  # Pause de 5 secondes avant de réessayer.
            return None
        except BinanceOrderException as e:  # Capture des erreurs liées à l'ordre lui-même.
            logger.error(f"Erreur dans l'ordre : {e}")  # Log une erreur liée à l'ordre.
            return None
        except Exception as e:  # Capture d'erreurs inattendues.
            logger.error(f"Erreur inattendue : {e}")  # Log une erreur inattendue.
            return None

    # Méthode pour exécuter un trade (achat ou vente) en fonction d'un signal et de la taille de la position.
    def execute_trade(self, signal, position_size):
        asset_balance = self.get_asset_balance()  # Obtient le solde de l'actif (par exemple, BTC).
        usdt_balance = self.get_account_balance()  # Obtient le solde en USDT.

        if signal == 'BUY':  # Si le signal est un achat (BUY).
            if usdt_balance < self.min_notional:  # Vérifie que le solde en USDT est suffisant.
                logger.warning("Solde USDT insuffisant pour acheter.")  # Log si le solde est insuffisant.
                return
            quantity = self.calculate_quantity(position_size)  # Calcule la quantité à acheter.
            if quantity >= self.step_size:  # Vérifie que la quantité est suffisante pour passer un ordre.
                self.place_order('BUY', quantity, stop_loss=0.02, take_profit=0.04)  # Place un ordre d'achat avec stop-loss et take-profit.
            else:
                logger.warning("Quantité calculée inférieure au minimum requis.")  # Log si la quantité est trop faible.
        elif signal == 'SELL' and asset_balance >= self.step_size:  # Si le signal est une vente (SELL) et que le solde est suffisant.
            quantity = asset_balance  # Utilise le solde total de l'actif pour la vente.
            quantity = float(round(quantity, int(round(-math.log(self.step_size, 10), 0))))  # Ajuste la quantité au step size.
            self.place_order('SELL', quantity)  # Place un ordre de vente.
        else:
            logger.info("Aucune action requise.")  # Log si aucune action n'est requise.
