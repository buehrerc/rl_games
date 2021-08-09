import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from interfaces import Player, Game


class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        return np.random.choice(possible_actions)

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class MiniMaxPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        raise NotImplementedError()


class DeepQPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        raise NotImplementedError()


class Connect4(Game):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.WINNING_NUMBER = 4
        self.BOARD_ROW, self.BOARD_COL = 6, 7
        self.board = np.zeros((self.BOARD_ROW, self.BOARD_COL), dtype=int)

    def _possible_actions(self):
        """Returns all possible action based on the current board state"""
        return [idx_col for idx_col, col in enumerate(self.board.T) if len(np.where(col == 0)[0]) > 0]

    def _update_board(self, chosen_action, by_player):
        """Function puts the token to the lowest free spot in the picked column"""
        picked_column = self.board[:, chosen_action]
        # Determine last cell which has value zero and set the player token there
        picked_column[np.where(picked_column == 0)[0][-1]] = self.playerSymbol[by_player.name]
        # Insert the picked column back into the board
        self.board[:, chosen_action] = picked_column

    def _check_on_board(self, direction):
        """
        Checks whether in the current direction all values are the same (-1 or 1)
        :param direction: Current slice of the board which is inspected
        :return: True if a winner was found, else False
        """
        if np.all(direction == self.playerSymbol[self.p1.name]):
            self.winner = self.p1
            return True
        elif np.all(direction == self.playerSymbol[self.p2.name]):
            self.winner = self.p2
            return True
        else:
            return False

    def _is_finished(self):
        """Check whether game is finished"""
        # Implementation is not optimal, since most of the columns are checked multiple times!
        for i, j in itertools.product(range(self.BOARD_ROW-self.WINNING_NUMBER+1),
                                      range(self.BOARD_COL-self.WINNING_NUMBER+1)):
            sub_board = self.board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER]
            # Horizontal
            for row in sub_board:
                if self._check_on_board(row) is True:
                    return True
            # Vertical
            for col in sub_board.transpose():
                if self._check_on_board(col) is True:
                    return True
            # Diagonal
            for diag in [np.diagonal(sub_board), np.diagonal(np.flip(sub_board, axis=1))]:
                if self._check_on_board(diag) is True:
                    return True
            # Tie
            if len(self._possible_actions()) == 0:
                self.winner = False
                return True
        return False

    def _match_summary(self):
        """
        Function determines the winner of the game and returns a nicely formatted board
        :return: [The name of the winning player or "Tie" if not winner, nicely formatted board_state]
        """
        vline = "--------------------------\n"
        output_string = vline
        winner_name = 'Tie' if self.winner is False else self.winner.name
        for row in self.board.astype(str):
            row = np.where(row == '1', 'O', row)
            row = np.where(row == '-1', 'X', row)
            row_formatted = np.where(row == '0', ' ', row)
            output_string += ' | '.join(row_formatted) + '\n'
        output_string += vline
        return winner_name, output_string


if __name__ == '__main__':
    p1_ = RandomPlayer('p1')
    p2_ = RandomPlayer('p2')

    log = list()
    for i in tqdm(range(10)):
        game = Connect4(p1_, p2_)
        winner_, _ = game.play()
        log.append(winner_)
        game = Connect4(p2_, p1_)
        winner_, _ = game.play()
        log.append(winner_)
    print(pd.Series(log).value_counts() / len(log))
