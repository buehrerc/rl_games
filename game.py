"""
This file holds different kind of Games
- TicTacToe
- Connect 4
"""
import itertools
import numpy as np
from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.playerSymbol = {p1.name: -1,
                             p2.name: 1}
        self.winner = None

        # Forward the assigned symbol to the players
        self.p1.set_symbol(self.playerSymbol[p1.name])
        self.p2.set_symbol(self.playerSymbol[p2.name])

    @abstractmethod
    def _possible_actions(self):
        """Returns all possible action based on the current board state"""
        pass

    @abstractmethod
    def _update_board(self, chosen_action, by_player):
        """Function puts the token to the lowest free spot in the picked column"""
        pass

    @abstractmethod
    def _is_finished(self):
        """Check whether game is finished"""
        pass

    @abstractmethod
    def _match_summary(self):
        """
        Function determines the winner of the game and returns a nicely formatted board
        :return: [The name of the winning player or "Tie" if not winner, nicely formatted board_state]
        """
        pass

    @abstractmethod
    def play(self):
        """
        Function runs the actual game and give each player the board state and possible actions in each round.
        :return: The name of the winning player or "TIE" if no winner.
        """
        pass


class TicTacToe(Game):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.BOARD_ROW, self.BOARD_COL = 3, 3
        self.board = np.zeros((self.BOARD_ROW, self.BOARD_COL), dtype=int)

    def _possible_actions(self):
        """Returns the possible actions on the board"""
        return np.where(self.board.reshape(self.BOARD_ROW*self.BOARD_COL) == 0)[0].tolist()

    def _update_board(self, action, player):
        """Updates board with the action that was taken by the player"""
        tmp_board = self.board.reshape(self.BOARD_ROW * self.BOARD_COL).copy()
        tmp_board[action] = self.playerSymbol[player.name]
        self.board = tmp_board.reshape(self.BOARD_ROW, self.BOARD_COL)

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
        # Horizontal
        for row in self.board:
            if self._check_on_board(row) is True:
                return True
        # Vertical
        for col in self.board.transpose():
            if self._check_on_board(col) is True:
                return True
        # Diagonal
        for diag in [np.diagonal(self.board), np.diagonal(np.flip(self.board, axis=1))]:
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
        separator = "---------\n"
        output_string = str()
        winner_name = 'Tie' if self.winner is False else self.winner.name
        for row in self.board.astype(str):
            row = np.where(row == '1', 'O', row)
            row = np.where(row == '-1', 'X', row)
            row_formatted = np.where(row == '0', ' ', row)
            output_string += ' | '.join(row_formatted) + '\n' + separator
        return winner_name, output_string

    def play(self):
        """
        Function runs the actual game and give each player the board state and possible actions in each round.
        :return: The name of the winning player or "TIE" if no winner.
        """
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
            # Let Player 2 take action
            player_action = self.p2.choose_action(self.board, actions)
            # Update board accordingly
            self._update_board(player_action, self.p2)
            # Check whether game is finished
            if self._is_finished():
                break
        winner_name, final_board_state = self._match_summary()
        # Give feedback to the players about the outcome of the match
        self.p1.receive_feedback(winner_name)
        self.p2.receive_feedback(winner_name)
        return winner_name, final_board_state


class Connect4(Game):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.WINNING_NUMBER = 4
        self.BOARD_ROW, self.BOARD_COL = 6, 7
        self.board = np.zeros((self.BOARD_ROW, self.BOARD_COL), dtype=int)

    def _possible_actions(self):
        """Returns all possible action based on the current board state"""
        return [idx_col for idx_col, col in enumerate(self.board.T) if any(col == 0)]

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
        # Horizontal
        for i, j in itertools.product(range(self.BOARD_ROW-self.WINNING_NUMBER+1), range(self.BOARD_COL)):
            if self._check_on_board(self.board[i:i+self.WINNING_NUMBER, j]) is True:
                return True
        # Vertical
        for i, j in itertools.product(range(self.BOARD_ROW), range(self.BOARD_COL - self.WINNING_NUMBER + 1)):
            if self._check_on_board(self.board[i, j:j + self.WINNING_NUMBER]) is True:
                return True
        # Diagonal
        for i, j in itertools.product(range(self.BOARD_ROW-self.WINNING_NUMBER+1),
                                      range(self.BOARD_COL-self.WINNING_NUMBER+1)):
            sub_board = self.board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER]
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

    def play(self):
        """
        Function runs the actual game and give each player the board state and possible actions in each round.
        :return: The name of the winning player or "TIE" if no winner.
        """
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
            # Let Player 2 take action
            player_action = self.p2.choose_action(self.board, actions)
            # Update board accordingly
            self._update_board(player_action, self.p2)
            # Check whether game is finished
            if self._is_finished():
                break

        winner_name, final_board_state = self._match_summary()
        # Give feedback to the players about the outcome of the match
        self.p1.receive_feedback(winner_name)
        self.p2.receive_feedback(winner_name)
        return winner_name, final_board_state
