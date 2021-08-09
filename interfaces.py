from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        pass

    @abstractmethod
    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        pass


class Game(ABC):
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.playerSymbol = {p1.name: -1,
                             p2.name: 1}
        self.winner = None

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
