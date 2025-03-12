import sqlite3

# Fonction pour créer la connexion à la base de données
def create_connection():
    conn = sqlite3.connect('market_data.db')  # Connexion à une base de données SQLite.
    return conn

# Fonction pour créer les tables dans la base de données si elles n'existent pas encore
def create_tables():
    conn = create_connection()
    cursor = conn.cursor()
    # Créer une table pour stocker les données de marché si elle n'existe pas déjà
    cursor.execute('''CREATE TABLE IF NOT EXISTS market_data (
                        symbol TEXT,
                        timestamp TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL
                      )''')
    conn.commit()  # Enregistre les modifications dans la base de données
    conn.close()  # Ferme la connexion à la base de données

# Fonction pour insérer des données de marché dans la table
def insert_market_data(data):
    conn = create_connection()
    cursor = conn.cursor()
    # Insertion des données de marché dans la table
    cursor.executemany('''INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                          VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)''', data)
    conn.commit()  # Enregistre les modifications dans la base de données
    conn.close()  # Ferme la connexion à la base de données
