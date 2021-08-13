"""
This File holds different kind of Players for TicTacToe
- HumanPlayer (Human Input based Player)
- MinimaxPlayer (Minimax plus Alpha-Beta Pruning based Player)
- MCTSPlayer (Monte Carlo Tree Search based Player)
- QPlayer (Q-Learning based Player)
- DQNPlayer (Deep Q-Learning based Player)
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from players import BasePlayer, BaseMinimaxPlayer, BaseHumanPlayer, RandomPlayer
from neural_networks import TTTFlatNetwork, TTTConvNetwork

import torch.nn as nn
import torch.optim as optim


class HumanPlayer(BaseHumanPlayer):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def _print_board(board):
        """Prints a nicely formatted RL_games board."""
        separator = "---------\n"
        output_string = str()
        for row in board.astype(str):
            row = np.where(row == '1', 'O', row)
            row = np.where(row == '-1', 'X', row)
            row_formatted = np.where(row == '0', ' ', row)
            output_string += ' | '.join(row_formatted) + '\n' + separator
        print(output_string)


class MinimaxPlayer(BaseMinimaxPlayer):
    """Minimax Player with Alpha-Beta pruning"""
    def __init__(self, name, depth_limit=7):
        super().__init__(name, depth_limit=depth_limit)

    def _get_utility_score(self, board):
        """
        Returns the utility score for the current board state, according to the following scores:
        - 3 symbol in the same direction -> 1.0
        - 2 symbol + 1 empty in the same direction -> 0.2
        :param board:
        :return:
        """
        utility = 0
        possible_directions = [row for row in board] + \
                              [col for col in board.T] + \
                              [np.diagonal(board), np.diagonal(np.flip(board, axis=1))]
        for direction in possible_directions:
            if np.sum(direction == self.symbol) == 3:
                utility += 1
            elif np.sum(direction == self.symbol) == 2 and np.sum(direction == 0) == 1:
                utility += 0.2
        return utility

    @staticmethod
    def _perform_action(board, action, symbol):
        """Updates board with the action that was taken by the player"""
        tmp_board = board.flatten().copy()
        tmp_board[action] = symbol
        return tmp_board.reshape(*board.shape)

    @staticmethod
    def _is_finished(board):
        """Check whether game is finished"""
        # Check whether there is a winner
        possible_directions = [row for row in board] + \
                              [col for col in board.transpose()] + \
                              [np.diagonal(board), np.diagonal(np.flip(board, axis=1))]
        for direction in possible_directions:
            if np.all(direction == '1') or np.all(direction == '-1'):
                return True
        # Check whether it is a tie
        if len(np.where(board == 0)[0]) == 0:
            return True
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
            board_plus_action = self._perform_action(board, action, self.symbol)
            # Remove the action from the possible actions
            action_removed = possible_actions[:i] + possible_actions[i+1:]
            # Explore the possible action
            _, returned_utility = self._minimize(board_plus_action, action_removed, depth_limit-1, alpha, beta)
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
            board_plus_action = self._perform_action(board, action, self.symbol * -1)
            # Remove the action from the possible actions
            action_removed = possible_actions[:i] + possible_actions[i+1:]
            # Explore the possible action
            _, returned_utility = self._maximize(board_plus_action, action_removed, depth_limit-1, alpha, beta)
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

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        action, _ = self._maximize(board, possible_actions, self.depth_limit, self.min_alpha, self.max_beta)
        return action

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # No implementation needed since player is not a learning agent.
        pass


class QPlayer(BasePlayer):
    """
    Q-Learning based Player
    For each possible board state a dictionary is maintained. For each possible action in the board state, a Q value
    is updated.
    """
    def __init__(self, name, epsilon=0.3, alpha=0.2, gamma=0.9):
        """
        :param name: Name of the player
        :param epsilon: Exploration Rate
        :param alpha: Learning Rate
        :param gamma: Gamma for decay
        """
        super().__init__(name=name)
        self.qtable = dict()  # The board states and its respective possible actions are saved in here.
        self.ex_rate = epsilon
        self.lr = alpha
        self.decay_gamma = gamma
        self.game_history = list()

    @staticmethod
    def _hash_board_state(board):
        """Hashes current board state into a string"""
        return ''.join(board.flatten().astype(str))

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: 3x3 matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        # Hash current board state
        board_hash = self._hash_board_state(board)
        # If the current board state is not in qtable -> add it and initialize all actions with Q value 0
        if board_hash not in self.qtable.keys():
            self.qtable[board_hash] = dict(zip(possible_actions, np.zeros(len(possible_actions))))
        # Do exploration with probability of ex_rate
        if np.random.rand() <= self.ex_rate:
            action = np.random.choice(possible_actions)
        # Otherwise, pick action with highest Q value
        # Break ties randomly
        else:
            q_values = pd.Series(self.qtable[board_hash])
            possible_actions = q_values[q_values == q_values.max()].index
            action = np.random.choice(possible_actions)
        self.game_history.append((board_hash, action))
        return action

    def receive_feedback(self, winner):
        """
        Backpropagates the reward to the different actions taken
        q_t = q_t + learning_rate * (q_(t+1)*decay_gamma - q_t)
        :param winner: name of the round's winner
        """
        if winner == self.name:
            reward = 10
        elif winner == 'Tie':
            reward = 5
        else:
            reward = -10
        # Back Propagate
        for round_ in reversed(self.game_history):
            current_q_value = self.qtable[round_[0]][round_[1]]
            current_q_value += self.lr * (self.decay_gamma * reward - current_q_value)
            reward = current_q_value
            self.qtable[round_[0]][round_[1]] = current_q_value
            self.game_history = list()

    def store_policy(self, file_name):
        """Saves current policy"""
        df = pd.DataFrame(self.qtable).T
        df.index = df.index.set_names('index')
        df.to_csv("./policies/{}.csv".format(file_name))

    def load_policy(self, file_name):
        """Loads policy in file_name"""
        df = pd.read_csv("./policies/{}.csv".format(file_name))
        df.set_index('index', inplace=True)
        df.columns = df.columns.astype(int)
        self.qtable = df.to_dict('index')


class DQNPlayer(BasePlayer):
    def __init__(self, name, model, epsilon=0, gamma=0.9, lr=0.01):
        super().__init__(name)
        # Initialize all the Player parameters
        self.ex_rate = epsilon
        self.decay_gamma = gamma
        self.lr = lr
        self.reward = {'win': 100,
                       'tie': 30,
                       'lose': -100,
                       'illegal': torch.tensor(-1000.0)}
        self.game_history = list()

        # Initialize all the PyTorch parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()

    def _convert_to_tensor(self, board):
        board = torch.tensor(board)
        if self.model.architecture == 'conv':
            empty = torch.zeros(board.shape).masked_scatter_((board == 0), torch.ones(board.shape)).view(1, 3, 3)
            own_moves = torch.zeros(board.shape).masked_scatter_((board == self.symbol), torch.ones(board.shape)).view(1, 3, 3)
            op_moves = torch.zeros(board.shape).masked_scatter_((board == self.symbol * -1), torch.ones(board.shape)).view(1, 3, 3)
            return torch.stack((empty, own_moves, op_moves))
        elif self.model.architecture == 'flat':
            return board.flatten().type(torch.float32)

    def _illegal_actions(self, board):
        """Returns the possible actions on the board"""
        if self.model.architecture == 'conv':
            return torch.where(board[0] == 0)[0].tolist()
        elif self.model.architecture == 'flat':
            return torch.where(board != 0)[0].tolist()

    def choose_action(self, board, possible_actions):
        """
        Function chooses best action based on provided board state and possible actions.
        :param board: matrix, representing the RL_games field
        :param possible_actions: possible fields to put next symbol
        :return: chosen action
        """
        tensor_board = self._convert_to_tensor(board)
        # Do exploration with probability of ex_rate
        if np.random.rand() <= self.ex_rate:
            action = np.random.choice(possible_actions)
        # Otherwise, pick action with highest Q value
        # Break ties randomly
        else:
            while True:
                #with torch.no_grad():
                output = self.model(tensor_board)
                action = int(np.argmax(output.detach().numpy()))
                # If the network chooses a wrong action, it will be punished immediately
                if action not in possible_actions:
                    self.model.train()
                    # Clear previously calculated gradients
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    # Compute the expected output
                    expected_output = torch.zeros(9)
                    # TODO: Is this the correct Q-Value for the possible actions?
                    expected_output[possible_actions] = 5.0
                    # Compute the expected output and its corresponding loss
                    expected_output = torch.where(expected_output == 0, self.reward['illegal'], expected_output)
                    loss = self.loss(output, expected_output)
                    # Back propagate to calculate the gradients
                    loss.backward()
                    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # update parameters
                    self.optimizer.step()
                # Else the action will be chosen and returned
                else:
                    break
        self.game_history.append((tensor_board, action))
        return action

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # Compute the reward
        if winner == self.name:
            reward = self.reward['win']
        elif winner == 'Tie':
            reward = self.reward['tie']
        else:
            reward = self.reward['lose']

        self.model.train()
        for board_state, action in reversed(self.game_history):
            # Clear previously calculated gradients
            self.model.zero_grad()
            self.optimizer.zero_grad()
            # Get model predictions for the board_state
            output = self.model(board_state)
            # Compute the expected output
            expected_output = torch.zeros(9)
            expected_output[action] = self.lr * (self.decay_gamma * reward)
            # For the illegal moves, the negative reward is set
            expected_output[self._illegal_actions(board_state)] = self.reward['illegal']
            # Compute the expected output and its corresponding loss
            loss = self.loss(output, expected_output)
            # Back propagate to calculate the gradients
            loss.backward()
            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # update parameters
            self.optimizer.step()
        # Reset game history
        self.game_history = list()

    def store_policy(self, name):
        """Saves the networks weights."""
        torch.save(self.model.state_dict(), r'./policies/{}.pt'.format(name))

    def load_policy(self, name):
        self.model.load_state_dict(torch.load(r'./policies/{}.pt'.format(name)))


if __name__ == '__main__':
    from game import TicTacToe
    """
    Training Documentation
    QPlayer('p1', alpha=0.2, epsilon=0.2, gamma=0.9) on RandomPlayer for 20000 rounds
    QPlayer('p1', alpha=0.2, epsilon=0.1, gamma=0.9) on MinimaxPlayer(depth_limit=5) for 1000 rounds
    """
    p1_ = DQNPlayer('p1', TTTConvNetwork(), epsilon=0.4, gamma=0.9, lr=0.01)
    p2_ = QPlayer('p2', epsilon=0, alpha=0, gamma=0.9)
    p2_.load_policy('tictactoe_q_policy')
    # p2_ = MinimaxPlayer('p2', depth_limit=5)

    print('training...')
    log_round = list()
    log_total = dict()
    for i in tqdm(range(10000)):
        game = TicTacToe(p1_, p2_)
        winner_, _ = game.play()
        log_round.append(winner_)
        game = TicTacToe(p2_, p1_)
        winner_, _ = game.play()
        log_round.append(winner_)
        if i % 100 == 0 and i != 0:
            current_stats = pd.Series(log_round).value_counts()/len(log_round)
            log_total[i] = current_stats.to_dict()
            print(current_stats)
    p1_.store_policy('tictactoe_dqn_policy')
    print('finished')