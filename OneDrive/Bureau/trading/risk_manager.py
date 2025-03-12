# Importation des bibliothèques nécessaires pour la gestion des risques.
import pandas as pd  # Utilisé pour manipuler les données, notamment les tableaux de données (DataFrames).
import numpy as np  # Utilisé pour les calculs mathématiques et la manipulation de tableaux numériques.
from config import MAX_DRAWDOWN, RISK_PER_TRADE, MAX_RISK_PER_POSITION  # Import des paramètres de gestion des risques depuis le fichier de configuration.
from trader import get_account_balance, get_trade_history  # Import de fonctions pour obtenir le solde du compte et l'historique des transactions.
from logger import setup_logger  # Importation du logger pour enregistrer les événements importants.
from datetime import datetime  # Utilisé pour obtenir l'heure et la date actuelle.

# Configuration du logger pour enregistrer les événements et erreurs
logger = setup_logger(__name__)

# Définition de la classe RiskManager pour gérer les risques dans les opérations de trading.
class RiskManager:
    def __init__(self):
        # Initialisation des attributs de la classe
        self.account_balance = get_account_balance()  # Obtient le solde actuel du compte.
        self.equity_curve = self._load_equity_curve()  # Charge l'équity curve (évolution du capital).
        self.max_drawdown_allowed = MAX_DRAWDOWN  # Limite maximale de drawdown (perte autorisée).
        self.risk_per_trade = RISK_PER_TRADE  # Pourcentage de risque à prendre par trade.
        self.max_risk_per_position = MAX_RISK_PER_POSITION  # Risque maximal par position (en pourcentage du capital).

    # Méthode pour charger ou initialiser l'équity curve.
    def _load_equity_curve(self):
        """
        Charge ou initialise l'équity curve du compte pour calculer le drawdown.
        """
        try:
            # Tente de charger l'équity curve à partir d'un fichier CSV.
            equity_data = pd.read_csv('data/equity_curve.csv', parse_dates=['timestamp'])
            logger.info("Équity curve chargée avec succès.")
        except FileNotFoundError:  # Si le fichier n'existe pas, initialise un nouveau tableau vide.
            equity_data = pd.DataFrame(columns=['timestamp', 'equity'])
            logger.warning("Aucune équity curve trouvée, initialisation d'une nouvelle.")
        return equity_data  # Retourne les données d'équity.

    # Méthode pour mettre à jour l'équity curve avec le solde actuel.
    def update_equity_curve(self):
        """
        Met à jour l'équity curve avec le solde actuel du compte.
        """
        # Crée une nouvelle entrée avec le solde actuel et la date actuelle.
        new_entry = {
            'timestamp': datetime.utcnow(),  # Timestamp de la mise à jour (heure UTC).
            'equity': self.account_balance  # Solde actuel du compte.
        }
        # Ajoute la nouvelle entrée à l'équity curve et sauvegarde dans un fichier CSV.
        self.equity_curve = self.equity_curve.append(new_entry, ignore_index=True)
        self.equity_curve.to_csv('data/equity_curve.csv', index=False)
        logger.info("Équity curve mise à jour.")

    # Méthode pour calculer le drawdown actuel basé sur l'évolution du capital.
    def calculate_current_drawdown(self):
        """
        Calcule le drawdown actuel basé sur l'équity curve.
        """
        if self.equity_curve.empty:  # Si l'équity curve est vide, on ne peut pas calculer le drawdown.
            logger.warning("Équity curve vide, impossible de calculer le drawdown.")
            return 0.0  # Retourne 0 si on ne peut rien calculer.

        # Calcul du drawdown en comparant l'équity actuelle au maximum atteint.
        equity_values = self.equity_curve['equity']
        max_equity = equity_values.cummax()  # Le maximum historique de l'équity.
        drawdowns = (equity_values - max_equity) / max_equity  # Calcul du drawdown.
        current_drawdown = drawdowns.iloc[-1]  # Récupère le dernier drawdown (celui du moment).
        logger.info(f"Drawdown actuel : {current_drawdown:.2%}")  # Log le drawdown actuel.
        return abs(current_drawdown)  # Retourne la valeur absolue du drawdown.

    # Méthode pour vérifier si le drawdown dépasse le maximum autorisé.
    def check_drawdown(self):
        """
        Vérifie si le drawdown actuel dépasse le maximum autorisé.
        """
        current_drawdown = self.calculate_current_drawdown()  # Calcul du drawdown actuel.
        if current_drawdown >= self.max_drawdown_allowed:  # Si le drawdown dépasse la limite.
            logger.warning("Le drawdown maximal autorisé a été atteint ou dépassé.")
            return False  # Retourne False pour indiquer que le drawdown est trop élevé.
        return True  # Retourne True si le drawdown est en dessous de la limite.

    # Méthode pour calculer la taille de la position à prendre en fonction du risque.
    def calculate_position_size(self, stop_loss_percentage):
        """
        Calcule la taille de position en fonction du risque par trade et du stop-loss.
        """
        # Calcul du montant du capital à risquer sur ce trade.
        risk_amount = self.account_balance * self.risk_per_trade
        # Calcul de la taille de la position basée sur le stop-loss.
        position_size = risk_amount / stop_loss_percentage

        # Limite la taille de la position au risque maximal autorisé par position.
        max_position_size = self.account_balance * self.max_risk_per_position
        position_size = min(position_size, max_position_size)  # Prend la plus petite valeur entre la taille calculée et la limite.

        logger.info(f"Taille de position calculée : {position_size:.4f} USDT")  # Log la taille de position calculée.
        return position_size  # Retourne la taille de la position.

    # Méthode pour calculer la Value at Risk (VaR) du portefeuille.
    def calculate_value_at_risk(self, confidence_level=0.95):
        """
        Calcule la Value at Risk (VaR) du portefeuille.
        """
        # Calcul des rendements quotidiens à partir de l'équity curve.
        returns = self.equity_curve['equity'].pct_change().dropna()
        if returns.empty:  # Si les rendements sont vides, on ne peut pas calculer la VaR.
            logger.warning("Pas de données de retour pour calculer la VaR.")
            return 0.0  # Retourne 0 si on ne peut pas calculer.

        # Moyenne et écart type des rendements.
        mean_return = returns.mean()
        std_dev = returns.std()
        # Calcul de la VaR en utilisant une distribution normale (95% de confiance par défaut).
        var = norm.ppf(1 - confidence_level, mean_return, std_dev) * self.account_balance
        logger.info(f"Value at Risk (VaR) à {confidence_level:.0%} : {var:.2f} USDT")  # Log la VaR calculée.
        return abs(var)  # Retourne la valeur absolue de la VaR.

    # Méthode pour calculer le ratio de Sharpe du portefeuille.
    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        """
        Calcule le ratio de Sharpe du portefeuille.
        """
        # Calcul des rendements quotidiens.
        returns = self.equity_curve['equity'].pct_change().dropna()
        if returns.empty:  # Si les rendements sont vides, on ne peut pas calculer le ratio de Sharpe.
            logger.warning("Pas de données de retour pour calculer le ratio de Sharpe.")
            return 0.0  # Retourne 0 si on ne peut pas calculer.

        # Calcul des rendements excédentaires en soustrayant le taux sans risque (par jour).
        excess_returns = returns - risk_free_rate / 252  # 252 jours de trading par an.
        # Calcul du ratio de Sharpe (rendement excédentaire divisé par la volatilité).
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        logger.info(f"Ratio de Sharpe : {sharpe_ratio:.2f}")  # Log le ratio de Sharpe.
        return sharpe_ratio  # Retourne le ratio de Sharpe.

    # Méthode pour évaluer les risques et décider si le trading doit continuer.
    def assess_risk(self):
        """
        Évalue le risque actuel et décide si le trading doit continuer.
        """
        if not self.check_drawdown():  # Si le drawdown est trop élevé, arrêter le trading.
            logger.warning("Trading arrêté en raison d'un drawdown excessif.")
            return False  # Retourne False pour indiquer que le trading doit s'arrêter.

        # Calcul de la VaR et vérification si elle dépasse le drawdown maximal.
        var = self.calculate_value_at_risk()
        if var > self.account_balance * self.max_drawdown_allowed:
            logger.warning("VaR dépasse le drawdown maximal autorisé.")
            return False  # Retourne False si la VaR dépasse la limite.

        return True  # Retourne True si tout est en ordre.

    # Méthode pour mettre à jour le solde du compte.
    def update_account_balance(self):
        """
        Met à jour le solde du compte.
        """
        self.account_balance = get_account_balance()  # Met à jour le solde en récupérant les données depuis l'API de l'exchange.
        logger.info(f"Solde du compte mis à jour : {self.account_balance:.2f} USDT")  # Log le nouveau solde du compte.

    # Méthode pour effectuer les vérifications de gestion des risques avant de trader.
    def perform_risk_management(self, stop_loss_percentage):
        """
        Effectue toutes les vérifications de gestion des risques avant de placer un trade.
        """
        self.update_account_balance()  # Met à jour le solde du compte.
        self.update_equity_curve()  # Met à jour l'équity curve avec le nouveau solde.

        # Évalue le risque actuel pour décider si le trading peut continuer.
        if not self.assess_risk():
            return 0.0  # Si les risques sont trop élevés, aucune position ne sera prise.

        # Calcule la taille de la position en fonction du stop-loss et du risque par trade.
        position_size = self.calculate_position_size(stop_loss_percentage)
        return position_size  # Retourne la taille de la position à prendre.
