"""
This files holds all functions to train the different players
- TicTacToe
    + QPlayer
    + DQNPlayer
- Connect4
    + DQNPlayer
"""
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------------------------------
# FUNCTION TOOLBOX
def run_training_session(game, player1, player2, rounds):
    print('Start Training Session')
    training_log = list()
    for r in tqdm(range(rounds)):
        game = game(player1, player2)
        winner, _ = game.play()
        training_log.append(winner)
        game = game(player1, player2)
        winner, _ = game.play()
        training_log.append(winner)
    return pd.Series(training_log).value_counts()/len(training_log)


def train_tictactoe_QPlayer():
    from game import TicTacToe
    from tictactoe import QPlayer, MinimaxPlayer, RandomPlayer
    # 1st Training Session
    p1 = QPlayer('p1', alpha=0.2, epsilon=0.2, gamma=0.9)
    p2 = RandomPlayer('p2')
    print(run_training_session(TicTacToe, p1, p2, 20000))

    # 2nd Training Session
    p1.alpha = 0.1
    p2 = MinimaxPlayer('p2', depth_limit=5)
    print(run_training_session(TicTacToe, p1, p2, 1000))

    p1.store_policy(r'tictactoe_q_policy')


if __name__ == '__main__':
    train_tictactoe_QPlayer()