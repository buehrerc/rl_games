"""
This file holds all functions to train the different players
- TicTacToe
    + QPlayer
    + DQNPlayer
- Connect4
    + DQNPlayer
    + MCTSPlayer
"""
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------------------------------
# TICTACTOE
def run_tictactoe_training_session(player1, player2, rounds):
    from game import TicTacToe
    print('Start Training Session')
    training_log = list()
    for r in tqdm(range(rounds)):
        game = TicTacToe(player1, player2)
        winner, _ = game.play()
        training_log.append(winner)
        game = TicTacToe(player2, player1)
        winner, _ = game.play()
        training_log.append(winner)
        if r % 100 == 0 and r != 0:
            print(pd.Series(training_log).value_counts() / len(training_log))
    return pd.Series(training_log).value_counts()/len(training_log)


def train_tictactoe_QPlayer():
    from tictactoe import QPlayer, MinimaxPlayer, RandomPlayer
    # 1st Training Session
    p1 = QPlayer('p1', alpha=0.2, epsilon=0.2, gamma=0.9)
    p2 = RandomPlayer('p2')
    print(run_tictactoe_training_session(p1, p2, 20000))

    # 2nd Training Session
    p1.alpha = 0.1
    p2 = MinimaxPlayer('p2', depth_limit=5)
    print(run_tictactoe_training_session(p1, p2, 1000))
    p1.store_policy(r'tictactoe_q_policy')


def train_tictactoe_DQNPlayer():
    from tictactoe import DQNPlayer, QPlayer, RandomPlayer
    from neural_networks import TTTPolicyNetwork, TTTQNetwork
    # 1st Training Session
    p1 = DQNPlayer('p1', TTTPolicyNetwork(), TTTQNetwork(), epsilon=0.3, gamma=0.9, lr=0.01)
    p1.load_policy(r'tictactoe_dqn_policy')
    # p2 = RandomPlayer('p2')
    # print(run_tictactoe_training_session(p1, p2, 5000))

    # 2nd Training Session
    p1.epsilon = 0.1
    p2 = QPlayer('p1', alpha=0, epsilon=0, gamma=0.9)
    p2.load_policy(r'tictactoe_q_policy')
    print(run_tictactoe_training_session(p1, p2, 1000))
    p1.store_policy(r'tictactoe_dqn_policy')


# ---------------------------------------------------------------------------------------------------------------------
# CONNECT4
def run_connect4_training_session(player1, player2, rounds):
    from game import Connect4
    print('Start Training Session')
    training_log = list()
    for r in tqdm(range(rounds)):
        game = Connect4(player1, player2)
        winner, _ = game.play()
        training_log.append(winner)
        game = Connect4(player2, player1)
        winner, _ = game.play()
        training_log.append(winner)
    print('End Training Session')
    return pd.Series(training_log).value_counts()/len(training_log)


def train_connect4_DQNPlayer():
    from connect4 import DQNPlayer, MinimaxPlayer
    from neural_networks import C4PolicyNetwork, C4QNetwork
    # 1st Training Session
    p1 = DQNPlayer('p1', C4PolicyNetwork(), C4QNetwork(), epsilon=0.2, gamma=0.9, lr=0.05)
    p2 = MinimaxPlayer('p2', depth_limit=5)
    print(run_connect4_training_session(p1, p2, 100))
    p1.store_policy(r'connect4_dqn_policy')


def train_connect4_MCTS():
    from connect4 import MCTSPlayer, RandomPlayer
    p1 = MCTSPlayer('p1')
    p2 = RandomPlayer('p2')
    print(run_connect4_training_session(p1, p2, 50))
    p1.store_policy(r'connect4_mcts_policy')


if __name__ == '__main__':
    train_connect4_MCTS()
