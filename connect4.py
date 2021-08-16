"""
This File holds different kind of Players for Connect4
- MinimaxPlayer (Minimax plus Alpha-Beta Pruning based Player)
- DQNPlayer (Deep Q-Learning based Player)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from players import BasePlayer, BaseMinimaxPlayer, BaseHumanPlayer, RandomPlayer
from neural_networks import C4PolicyNetwork, C4QNetwork


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
    """Minimax Player with Alpha-Beta pruning"""
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
            if np.sum(direction == self.symbol) < 3 and np.sum(direction == self.symbol * -1) < 3:
                continue
            # Look for the section where scores could be assigned
            for i in range(len(direction-self.WINNING_NUMBER+1)):
                sub_direction = direction[i:i+self.WINNING_NUMBER]
                if np.sum(sub_direction == self.symbol) == 4:
                    utility += 4
                if np.sum(sub_direction == self.symbol) == 3 and np.sum(sub_direction == 0) == 1:
                    utility += 0.2

            # Look for the section where scores could be assigned
            for i in range(len(direction - self.WINNING_NUMBER + 1)):
                sub_direction = direction[i:i + self.WINNING_NUMBER]
                if np.sum(sub_direction == self.symbol * -1) == 4:
                    utility -= 4
                if np.sum(sub_direction == self.symbol * -1) == 3 and np.sum(sub_direction == 0) == 1:
                    utility -= 0.2
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
                              [np.diagonal(board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER])
                               for i, j in itertools.product(range(self.BOARD_ROW - self.WINNING_NUMBER + 1),
                                                             range(self.BOARD_COL - self.WINNING_NUMBER + 1))] + \
                              [np.diagonal(np.flip(board[i:i + self.WINNING_NUMBER, j:j + self.WINNING_NUMBER], axis=1))
                               for i, j in itertools.product(range(self.BOARD_ROW - self.WINNING_NUMBER + 1),
                                                             range(self.BOARD_COL - self.WINNING_NUMBER + 1))]
        for direction in possible_directions:
            if np.all(direction == 1) or np.all(direction == -1):
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
            # 1) Explore the possible action
            # 1.1) Update the board for the current action
            board_plus_action = self._perform_action(board.copy(), action, self.symbol * -1)
            # 1.2) Update possible actions according to taken action
            action_updated = self._possible_actions(board_plus_action)
            # 1.3) Explore the possible action
            _, returned_utility = self._maximize(board_plus_action, action_updated, depth_limit-1, alpha, beta)
            # 2) Update current best move
            if min_utility > returned_utility:
                min_utility = returned_utility
                move = action
            # 3) Update Beta value accordingly
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
    """
    Deep Q-Learning based Player
    Source: https://arxiv.org/pdf/1312.5602.pdf

    The Player holds a neural network for the Q-value prediction and a neural netowrk for the policy prediction.
    """
    def __init__(self, name, policy_model, q_model, epsilon=0.1, gamma=0.9, lr=0.01):
        """
        :param name: Name of the player
        :param policy_model: NN for Action policy
        :param q_model: NN for Q value prediction
        :param epsilon: Exploration Rate
        :param gamma: Reward decay
        :param lr: Learning Rate
        """
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
        self.q_model = q_model.to(self.device)
        self.q_optimizer = optim.AdamW(params=self.q_model.parameters(), lr=self.lr)
        self.q_loss = nn.MSELoss()
        self.policy_model = policy_model.to(self.device)
        self.policy_optimizer = optim.AdamW(params=self.policy_model.parameters(), lr=self.lr)
        self.policy_loss = nn.BCELoss()

    def _convert_to_tensor(self, board):
        board = torch.tensor(board)
        empty = torch.zeros(board.shape).masked_scatter_((board == 0), torch.ones(board.shape)).view(1, 6, 7)
        own_moves = torch.zeros(board.shape).masked_scatter_((board == self.symbol), torch.ones(board.shape)).view(1, 6, 7)
        op_moves = torch.zeros(board.shape).masked_scatter_((board == self.symbol * -1), torch.ones(board.shape)).view(1, 6, 7)
        return torch.stack((empty, own_moves, op_moves))

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
        else:
            while True:
                pred_policy = self.policy_model(tensor_board)
                action = int(np.argmax(pred_policy.detach().numpy()))
                # If the network chooses a wrong action, the policy network will be punished immediately
                if action not in possible_actions:
                    # Optimize the action policy network accordingly
                    self._optimize_policy_network(tensor_board, action, self.reward['illegal'])
                # Else the action will be chosen and returned
                else:
                    break
        self.game_history.append((tensor_board, action))
        return action

    @staticmethod
    def _legal_actions(board):
        """Finds all legal actions that could be taken on the board."""
        return [i for i, col in enumerate(board[0, 0].T) if any(col == 1)]

    def _compute_expected_policy(self, board_state, action, reward):
        """
        Computes the expected policy the policy network should produced.
        Currently, the expected policy is designed as follows:
        - if move lead to positive reward -> 1 at chosen move, 0 otherwise
        - if move lead to negative reward -> 1 at every legal move, 0 otherwise
        TODO: Compute the expected policy using MiniMax or MCTS
        """
        expected_output = torch.zeros(7)
        # Taken action lead to win
        if reward > 0:
            expected_output[action] = 1
        else:
            # For the remaining legal moves, the 1 is set
            expected_output[self._legal_actions(board_state)] = 1
            # 0 for the chosen action
            expected_output[action] = 0
        return expected_output

    def _optimize_policy_network(self, board_state, action, reward):
        """Updates the Action policy network"""
        # 1) Clear previously calculated gradients
        self.policy_model.zero_grad()
        self.policy_optimizer.zero_grad()
        # 2) Get model predictions for the board_state
        pred_policy = self.policy_model(board_state)
        # 3) Compute the expected output
        expected_policy = self._compute_expected_policy(board_state, action, reward)
        # 4) Compute the loss
        policy_loss = self.policy_loss(pred_policy, expected_policy)
        # 5) Backward propagate the loss
        policy_loss.backward()
        # 6) pdate parameters
        self.policy_optimizer.step()

    def _optimize_q_network(self, board_state, reward):
        """Updates the Q value network"""
        # 1) Clear previously calculated gradients
        self.q_model.zero_grad()
        self.q_optimizer.zero_grad()
        # 2) Get model predictions for the board_state
        pred_q_value = self.q_model(board_state)
        # 3) Compute the expected output
        expected_q_value = reward
        # 4) Compute the loss
        q_loss = self.q_loss(pred_q_value, expected_q_value)
        # 5) Backward propagate the loss
        q_loss.backward()
        # 6) pdate parameters
        self.q_optimizer.step()

    def receive_feedback(self, winner):
        """Incorporates feedback from the game round into the policy"""
        # Compute the reward
        if winner == self.name:
            reward = torch.tensor(self.reward['win'], dtype=torch.float32)
        elif winner == 'Tie':
            reward = torch.tensor(self.reward['tie'], dtype=torch.float32)
        else:
            reward = torch.tensor(self.reward['lose'], dtype=torch.float32)

        self.policy_model.train()
        self.q_model.train()
        for board_state, action in reversed(self.game_history):
            # 1) Optimize the policy network
            self._optimize_policy_network(board_state, action, reward)
            # 2) Optimize the Q network
            self._optimize_q_network(board_state, reward)
            # 3) Update Reward for the next round
            reward *= self.decay_gamma
        # Reset game history
        self.game_history = list()

    def store_policy(self, name):
        """Saves the both networks weights"""
        torch.save(self.q_model.state_dict(), r'./policies/{}_q.pt'.format(name))
        torch.save(self.policy_model.state_dict(), r'./policies/{}_p.pt'.format(name))

    def load_policy(self, name):
        """Loads policy and Q-value network from input name"""
        self.q_model.load_state_dict(torch.load(r'./policies/{}_q.pt'.format(name)))
        self.policy_model.load_state_dict(torch.load(r'./policies/{}_p.pt'.format(name)))


if __name__ == '__main__':
    from game import Connect4
    p1_ = DQNPlayer('p1', C4PolicyNetwork(), C4QNetwork())
    p2_ = MinimaxPlayer('p2', depth_limit=5)

    game = Connect4(p1_, p2_)
    _, board_ = game.play()
    print(board_)
