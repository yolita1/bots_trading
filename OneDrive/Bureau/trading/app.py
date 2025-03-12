from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Utiliser la variable globale `trading_data` que tu mets à jour dans trading_bot.py
trading_data = {}

@app.route('/')
def index():
    return render_template('index.html')

# Fonction pour envoyer des mises à jour de trading via WebSocket
def send_trading_updates():
    while True:
        socketio.emit('trading_update', trading_data)  # Envoie les nouvelles données à chaque mise à jour
        socketio.sleep(5)  # Envoie les mises à jour toutes les 5 secondes

# Démarrer un thread pour gérer l'envoi des mises à jour
def start_trading_updates():
    thread = threading.Thread(target=send_trading_updates)
    thread.start()

if __name__ == '__main__':
    start_trading_updates()
    socketio.run(app)
