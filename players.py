"""
This File holds different kind of Player interfaces
- BasePlayer
- BaseMinimaxPlayer
- BaseHumanPlayer
- RandomPlayer
"""

import numpy as np
from abc import ABC, abstractmethod


class BasePlayer(ABC):
    def __init__(self, name):
        self.name = name
        self.symbol = None

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

    def set_symbol(self, symbol):
        """Function is triggered by Game to give the player its symbol"""
        self.symbol = symbol


class BaseMinimaxPlayer(BasePlayer, ABC):
    """Minimax Player with Alpha-Beta pruning"""
    def __init__(self, name, depth_limit=7):
        super().__init__(name)
        self.depth_limit = depth_limit
        self.min_alpha, self.max_beta = float('-inf'), float('inf')

    @abstractmethod
    def _maximize(self, board, possible_actions, depth_limit, alpha, beta):
        """
        Maximize step of the Minimax algorithm
        :param board: current board state
        :param possible_actions: currently available actions
        :param depth_limit: stack limit
        :param alpha: alpha of alpha-beta pruning
        :param beta: beta of alpha-beta pruning
        :return: action to take
        """
        pass

    @abstractmethod
    def _minimize(self, board, possible_actions, depth_limit, alpha, beta):
        """
        Minimize step of the Minimax algorithm
        :param board: current board state
        :param possible_actions: currently available actions
        :param depth_limit: stack limit
        :param alpha: alpha of alpha-beta pruning
        :param beta: beta of alpha-beta pruning
        :return: action to take
        """
        pass

    @abstractmethod
    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        action, _ = self._maximize(board, possible_actions, self.depth_limit, self.min_alpha, self.max_beta)
        return action

    @abstractmethod
    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class BaseHumanPlayer(BasePlayer, ABC):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    @abstractmethod
    def _print_board(board):
        """Prints a nicely formatted RL_games board."""
        pass

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: 3x3 matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        self._print_board(board)
        while True:
            user_choice = int(input("Which Field?"))
            if user_choice in possible_actions:
                return user_choice
            else:
                print('Action not possible!')

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class RandomPlayer(BasePlayer):
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
