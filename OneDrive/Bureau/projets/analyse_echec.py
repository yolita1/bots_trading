import requests
import pandas as pd
import matplotlib.pyplot as plt
import chess
import chess.engine
from datetime import datetime

# Configuration
CHESS_COM_API = "https://api.chess.com/pub/player"
STOCKFISH_PATH = "stockfish"  # Télécharge Stockfish et indique le chemin ici

# Fonction pour récupérer les parties d'un utilisateur Chess.com
def get_games(username, year, month):
    url = f"{CHESS_COM_API}/{username}/games/{year}/{month}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['games']
    else:
        print(f"Erreur : Impossible de récupérer les parties pour {username}")
        return []

# Fonction pour analyser une partie avec Stockfish
def analyze_game(game, engine):
    board = chess.Board()
    mistakes = []
    blunders = []
    best_moves = []

    for move in game['moves'].split():
        # Analyser le coup joué
        board.push_san(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        best_move = info['pv'][0]
        best_moves.append(best_move)

        # Détecter les erreurs
        if 'score' in info:
            score = info['score'].relative.score()
            if abs(score) > 200:  # Erreur significative
                if abs(score) > 500:  # Grosse erreur (blunder)
                    blunders.append((move, best_move, score))
                else:
                    mistakes.append((move, best_move, score))

    return mistakes, blunders, best_moves

# Fonction pour générer des recommandations
def generate_recommendations(mistakes, blunders):
    recommendations = []

    if len(blunders) > 5:
        recommendations.append("Vous faites trop de grosses erreurs (blunders). Travaillez vos tactiques avec des exercices de puzzles.")

    if len(mistakes) > 10:
        recommendations.append("Vous faites des erreurs stratégiques. Étudiez les plans typiques de votre ouverture préférée.")

    if not recommendations:
        recommendations.append("Votre jeu est solide. Concentrez-vous sur l'amélioration de vos fins de partie.")

    return recommendations

# Fonction pour visualiser les résultats
def visualize_results(mistakes, blunders):
    # Graphique des erreurs par type
    labels = ['Mistakes', 'Blunders']
    counts = [len(mistakes), len(blunders)]
    plt.bar(labels, counts, color=['orange', 'red'])
    plt.title("Erreurs par Type")
    plt.ylabel("Nombre d'Erreurs")
    plt.show()

# Fonction principale
def main():
    username = input("Entrez votre nom d'utilisateur Chess.com : ")
    year = input("Entrez l'année à analyser (ex: 2023) : ")
    month = input("Entrez le mois à analyser (ex: 10) : ")

    # Récupérer les parties
    print("Récupération des parties en cours...")
    games = get_games(username, year, month)
    if not games:
        return

    # Initialiser Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Analyser chaque partie
    all_mistakes = []
    all_blunders = []
    for game in games:
        if game['rules'] == 'chess':  # Ignorer les variantes
            print(f"Analyse de la partie du {datetime.fromtimestamp(game['end_time'])}...")
            mistakes, blunders, _ = analyze_game(game, engine)
            all_mistakes.extend(mistakes)
            all_blunders.extend(blunders)

    # Fermer Stockfish
    engine.quit()

    # Générer des recommandations
    recommendations = generate_recommendations(all_mistakes, all_blunders)
    print("\nRecommandations pour vous améliorer :")
    for rec in recommendations:
        print(f"- {rec}")

    # Visualiser les résultats
    visualize_results(all_mistakes, all_blunders)

if __name__ == "__main__":
    main()