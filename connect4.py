"""
This File holds different kind of Players for Connect4
- MinimaxPlayer (Minimax plus Alpha-Beta Pruning based Player)
- DQNPlayer (Deep Q-Learning based Player)
"""
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from players import BasePlayer, BaseMinimaxPlayer, BaseHumanPlayer, RandomPlayer


class HumanPlayer(BaseHumanPlayer):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def _print_board(board):
        """Prints a nicely formatted RL_games board."""
        vline = "--------------------------\n"
        output_string = vline
        for row in board.astype(str):
            row = np.where(row == '1', 'O', row)
            row = np.where(row == '-1', 'X', row)
            row_formatted = np.where(row == '0', ' ', row)
            output_string += ' | '.join(row_formatted) + '\n'
        output_string += vline
        print(output_string)


class MinimaxPlayer(BaseMinimaxPlayer):
    def __init__(self, name, depth_limit):
        super().__init__(name, depth_limit=depth_limit)
        self.BOARD_ROW, self.BOARD_COL = 6, 7
        self.WINNING_NUMBER = 4

    @staticmethod
    def _possible_actions(board):
        """Returns all possible action based on the current board state"""
        return [idx_col for idx_col, col in enumerate(board.T) if len(np.where(col == 0)[0]) > 0]

    def _get_utility_score(self, board):
        """
        Returns the utility score for the current board state
        - 4 consecutive: 4
        - 3 consecutive: 0.2
        Note that the player's performance solely relies on the utility function
        """
        # TODO: Utility function should hold negative values as well and incorporates strategies
        utility = 0
        possible_directions = [row for row in board] + \
                              [col for col in board.T] + \
                              [np.diagonal(board, offset=offset)
                               for offset in range(-self.BOARD_ROW+1, self.BOARD_ROW)] + \
                              [np.diagonal(np.flip(board, axis=1), offset=offset)
                               for offset in range(-self.BOARD_ROW+1, self.BOARD_ROW)]
        for direction in possible_directions:
            # No utility can be found in directions which are shorter than 4
            if len(direction) < 4:
                continue
            # If there are less than 3 player symbols in the direction, no utility score will be assigned
            if np.sum(direction == self.symbol) < 3:
                continue
            # Look for the section where scores could be assigned
            for i in range(len(direction-self.WINNING_NUMBER+1)):
                sub_direction = direction[i:i+self.WINNING_NUMBER]
                if np.sum(sub_direction == self.symbol) == 4:
                    utility += 4
                if np.sum(sub_direction == self.symbol) == 3 and np.sum(sub_direction == 0) == 1:
                    utility += 0.2
        return utility

    @staticmethod
    def _perform_action(board, action, symbol):
        """Updates board with the action that was taken by the player"""
        picked_column = board[:, action]
        # Determine last cell which has value zero and set the player token there
        picked_column[np.where(picked_column == 0)[0][-1]] = symbol
        # Insert the picked column back into the board
        board[:, action] = picked_column
        return board

    def _is_finished(self, board):
        """Check whether game is finished"""
        # Check if there is a winner
        possible_directions = [board[i:i + self.WINNING_NUMBER, j]
                               for i, j in itertools.product(range(self.BOARD_ROW - self.WINNING_NUMBER + 1),
                                                             range(self.BOARD_COL))] + \
                              [board[i, j:j + self.WINNING_NUMBER]
                               for i, j in itertools.product(range(self.BOARD_ROW),
                                                             range(self.BOARD_COL - self.WINNING_NUMBER + 1))] + \
                              [board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER]
                               for i, j in itertools.product(range(self.BOARD_ROW - self.WINNING_NUMBER + 1),
                                                             range(self.BOARD_COL - self.WINNING_NUMBER + 1))]
        for direction in possible_directions:
            if np.all(direction == '1') or np.all(direction == '-1'):
                return True
        # Check if there is a Tie
        if len(np.where(board.flatten() == 0)[0]) == 0:
            return True
        # Game is not finished by now
        return False

    def _maximize(self, board, possible_actions, depth_limit, alpha, beta):
        # End Condition
        if depth_limit == 0 or len(possible_actions) == 0 or self._is_finished(board):
            return None, self._get_utility_score(board)
        # Explore the nodes
        max_utility, move = -1, None
        for i, action in enumerate(possible_actions):
            # 1. Explore the possible action
            # Update the board for the current action
            board_plus_action = self._perform_action(board.copy(), action, self.symbol)
            # Update possible actions according to taken action
            action_updated = self._possible_actions(board_plus_action)
            # Explore the possible action
            _, returned_utility = self._minimize(board_plus_action, action_updated, depth_limit-1, alpha, beta)
            # 2. Update current best move
            if max_utility < returned_utility:
                max_utility = returned_utility
                move = action
            # 3. Update Alpha value accordingly
            if alpha < returned_utility:
                alpha = returned_utility
                # Alpha-Beta Pruning Break Condition
                if alpha > beta:
                    break
        return move, max_utility

    def _minimize(self, board, possible_actions, depth_limit, alpha, beta):
        # End Condition
        if depth_limit == 0 or len(possible_actions) == 0 or self._is_finished(board):
            return None, self._get_utility_score(board)
        # Explore the nodes
        min_utility, move = 2, None
        for i, action in enumerate(possible_actions):
            # 1. Explore the possible action
            # Update the board for the current action
            board_plus_action = self._perform_action(board.copy(), action, self.symbol * -1)
            # Update possible actions according to taken action
            action_updated = self._possible_actions(board_plus_action)
            # Explore the possible action
            _, returned_utility = self._maximize(board_plus_action, action_updated, depth_limit-1, alpha, beta)
            # 2. Update current best move
            if min_utility > returned_utility:
                min_utility = returned_utility
                move = action
            # 3. Update Beta value accordingly
            if beta > returned_utility:
                beta = returned_utility
                # Alpha-Beta Pruning Break Condition
                if alpha > beta:
                    break
        return move, min_utility

    def _rearrange_possible_actions(self, actions):
        """
        Rearranges the ascendingly sorted list of actions to a center focused list:
        :param actions: [0, 1, 2, 3, 4, 5, 6]
        :return: [3, 4, 2, 5, 1, 6, 0]
        """
        if len(actions) == 1:
            return actions
        else:
            return [actions.pop(len(actions)//2)] + self._rearrange_possible_actions(actions)

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        # prefer actions which are in center
        possible_actions = self._rearrange_possible_actions(possible_actions)
        action, _ = self._maximize(board, possible_actions, self.depth_limit, self.min_alpha, self.max_beta)
        return action

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class DQNPlayer(BasePlayer):
    # Current equivalent to random player
    """
    Deep Q-Learning based Player
    Source: https://arxiv.org/pdf/1312.5602.pdf
    """
    def __init__(self, name, epsilon=1):
        """
        :param name: Name of the player
        :param epsilon: Exploration Rate
        """
        super().__init__(name)
        self.game_history = list()
        self.ex_rate = epsilon
        self.reward = {'win': 5,
                       'tie': 1,
                       'lose': -5,
                       'illegal': -10}

    def _hash_board_state(self, board):
        """Convert board into the input format for the NN"""
        empty_positions = np.array(board == 0, dtype=int)
        own_positions = np.array(board == self.symbol, dtype=int)
        op_positions = np.array((board != 0) & (board != self.symbol), dtype=int)
        return [own_positions, op_positions, empty_positions]

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        # Do exploration with probability of ex_rate
        if np.random.rand() <= self.ex_rate:
            action = np.random.choice(possible_actions)
        # Otherwise, pick action with highest Q value
        # Break ties randomly
        else:
            # NOTE: If the network chooses a wrong action, it will be punished immediately
            raise NotImplementedError()
        self.game_history.append((board, action))
        return action

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        pass


if __name__ == '__main__':
    from game import Connect4
    p1_ = MinimaxPlayer('p1', depth_limit=5)
    p2_ = HumanPlayer('p2')

    game = Connect4(p1_, p2_)
    game.play()

    # log = list()
    # for r in tqdm(range(10)):
    #     game = Connect4(p1_, p2_)
    #     winner_, _ = game.play()
    #     log.append(winner_)
    #     game = Connect4(p2_, p1_)
    #     winner_, _ = game.play()
    #     log.append(winner_)
    # print(pd.Series(log).value_counts() / len(log))
