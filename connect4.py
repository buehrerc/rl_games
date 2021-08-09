import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: 3x3 matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        pass

    @abstractmethod
    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        pass


class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: 3x3 matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        return np.random.choice(possible_actions)

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class MiniMaxPlayer(Player):
    pass


class DeepQPlayer(Player):
    pass


class Connect4:
    def __init__(self, p1, p2):
        self.WINNING_NUMBER = 4
        self.BOARD_ROW, self.BOARD_COL = 6, 7
        self.board = np.zeros((self.BOARD_ROW, self.BOARD_COL), dtype=int)
        self.p1, self.p2 = p1, p2
        self.playerSymbol = {p1.name: -1,
                             p2.name: 1}
        self.winner = None

    def play(self):
        while True:
            # Player 1's Turn
            # Get possible actions
            actions = self._possible_actions()
            # Let Player 1 take action
            player_action = self.p1.choose_action(self.board, actions)
            # Update board accordingly
            self._update_board(player_action, self.p1)
            # Check whether game is finished
            if self._is_finished():
                break

            # Player 2's Turn
            # Get possible actions
            actions = self._possible_actions()
            # Let Player 1 take action
            player_action = self.p1.choose_action(self.board, actions)
            # Update board accordingly
            self._update_board(player_action, self.p1)
            # Check whether game is finished
            if self._is_finished():
                break
        winner_name, final_board_state = self._match_summary()
        # Give feedback to the players about the outcome of the match
        self.p1.receive_feedback(winner_name)
        self.p2.receive_feedback(winner_name)
        return winner_name, final_board_state

    def _possible_actions(self):
        """Returns all possible action based on the current board state"""
        return [idx_col for idx_col, col in enumerate(self.board.T) if len(np.where(col == 0)[0]) > 0]

    def _update_board(self, chosen_action, player):
        """Function puts the token to the lowest free spot in the picked column"""
        picked_column = self.board[:, chosen_action]
        # Determine last cell which has value zero and set the player token there
        picked_column[np.where(picked_column == 0)[0][-1]] = self.playerSymbol[player.name]
        # Insert the picked column back into the board
        self.board[:, chosen_action] = picked_column

    def _is_finished(self):
        for i, j in itertools.product(range(self.BOARD_COL-self.WINNING_NUMBER),
                                      range(self.BOARD_ROW-self.WINNING_NUMBER)):
            sub_board = self.board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER]
            print('stop')
        return False

    def _match_summary(self):
        return None, None


if __name__ == '__main__':
    p1_ = RandomPlayer('p1')
    p2_ = RandomPlayer('p2')
    game = Connect4(p1_, p2_)
    game.play()
