# Importation des bibliothèques nécessaires
import smtplib  # Utilisé pour envoyer des emails via le protocole SMTP.
import asyncio  # Utilisé pour exécuter des tâches asynchrones (en parallèle).
import aiohttp  # Utilisé pour effectuer des requêtes HTTP de manière asynchrone.
from email.mime.text import MIMEText  # Utilisé pour formater le contenu des emails.
from config import (  # Importation des configurations nécessaires depuis le fichier de configuration.
    EMAIL, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SLACK_WEBHOOK_URL
)
from logger import setup_logger  # Importation du logger pour enregistrer les événements importants.

# Configuration du logger pour enregistrer les événements et erreurs
logger = setup_logger(__name__)

# Définition de la classe Notifier pour gérer l'envoi des notifications
class Notifier:
    def __init__(self):
        # Vérifie si la configuration pour l'envoi d'email est complète
        self.email_enabled = all([EMAIL, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT])
        # Vérifie si la configuration pour Telegram est complète
        self.telegram_enabled = all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID])
        # Vérifie si la configuration pour Slack est présente
        self.slack_enabled = SLACK_WEBHOOK_URL is not None

    # Méthode asynchrone pour envoyer un email
    async def send_email(self, subject, message):
        if not self.email_enabled:  # Si la configuration email est incomplète
            logger.warning("Configuration email incomplète. Notification par email non envoyée.")  # Log un avertissement.
            return
        try:
            # Formatage du contenu de l'email
            msg = MIMEText(message)
            msg['Subject'] = subject  # Sujet de l'email
            msg['From'] = EMAIL  # Adresse de l'expéditeur
            msg['To'] = EMAIL  # Adresse du destinataire (dans ce cas, même adresse)

            # Envoi de l'email via le serveur SMTP
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)  # Connexion au serveur SMTP
            server.starttls()  # Démarrage d'une connexion sécurisée
            server.login(EMAIL, EMAIL_PASSWORD)  # Authentification avec les identifiants
            server.sendmail(EMAIL, EMAIL, msg.as_string())  # Envoi de l'email
            server.quit()  # Fermeture de la connexion SMTP
            logger.info("Email de notification envoyé avec succès.")  # Log le succès de l'envoi de l'email.
        except Exception as e:  # Capture des erreurs éventuelles lors de l'envoi de l'email.
            logger.error(f"Erreur lors de l'envoi de l'email : {e}")  # Log l'erreur.

    # Méthode asynchrone pour envoyer une notification via Telegram
    async def send_telegram(self, message):
        if not self.telegram_enabled:  # Si la configuration Telegram est incomplète
            logger.warning("Configuration Telegram incomplète. Notification Telegram non envoyée.")  # Log un avertissement.
            return
        try:
            # Envoi de la requête HTTP POST à l'API Telegram pour envoyer le message
            async with aiohttp.ClientSession() as session:  # Crée une session HTTP asynchrone.
                url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'  # URL de l'API Telegram.
                payload = {
                    'chat_id': TELEGRAM_CHAT_ID,  # Identifiant du chat (utilisateur ou groupe).
                    'text': message  # Message à envoyer.
                }
                async with session.post(url, data=payload) as resp:  # Envoi de la requête POST.
                    if resp.status == 200:  # Si la requête réussit (code 200).
                        logger.info("Notification Telegram envoyée avec succès.")  # Log le succès de l'envoi.
                    else:
                        logger.error(f"Erreur lors de l'envoi de la notification Telegram : {resp.status}")  # Log une erreur si l'envoi échoue.
        except Exception as e:  # Capture des erreurs éventuelles lors de l'envoi du message Telegram.
            logger.error(f"Exception lors de l'envoi de la notification Telegram : {e}")  # Log l'erreur.

    # Méthode asynchrone pour envoyer une notification via Slack
    async def send_slack(self, message):
        if not self.slack_enabled:  # Si la configuration Slack est incomplète
            logger.warning("Configuration Slack incomplète. Notification Slack non envoyée.")  # Log un avertissement.
            return
        try:
            # Envoi de la requête HTTP POST au webhook Slack
            async with aiohttp.ClientSession() as session:  # Crée une session HTTP asynchrone.
                payload = {
                    'text': message  # Message à envoyer sur Slack.
                }
                async with session.post(SLACK_WEBHOOK_URL, json=payload) as resp:  # Envoi de la requête POST.
                    if resp.status == 200:  # Si la requête réussit (code 200).
                        logger.info("Notification Slack envoyée avec succès.")  # Log le succès de l'envoi.
                    else:
                        logger.error(f"Erreur lors de l'envoi de la notification Slack : {resp.status}")  # Log une erreur si l'envoi échoue.
        except Exception as e:  # Capture des erreurs éventuelles lors de l'envoi du message Slack.
            logger.error(f"Exception lors de l'envoi de la notification Slack : {e}")  # Log l'erreur.

    # Méthode asynchrone pour envoyer des notifications via tous les canaux configurés (email, Telegram, Slack)
    async def notify(self, subject, message):
        tasks = []  # Liste pour stocker les tâches asynchrones.
        if self.email_enabled:
            tasks.append(self.send_email(subject, message))  # Ajoute l'envoi de l'email à la liste des tâches.
        if self.telegram_enabled:
            tasks.append(self.send_telegram(message))  # Ajoute l'envoi de la notification Telegram à la liste des tâches.
        if self.slack_enabled:
            tasks.append(self.send_slack(message))  # Ajoute l'envoi de la notification Slack à la liste des tâches.
        if tasks:
            await asyncio.gather(*tasks)  # Exécute toutes les tâches en parallèle si au moins un canal est configuré.
        else:
            logger.warning("Aucun canal de notification n'est configuré.")  # Log un avertissement si aucun canal n'est configuré.

# Exemple d'utilisation pour tester les notifications
if __name__ == "__main__":
    notifier = Notifier()  # Instanciation de la classe Notifier.
    asyncio.run(notifier.notify("Test Notification", "Ceci est un test de notification."))  # Exécution d'un test pour envoyer une notification.
