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


class HumanPlayer(Player):
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


class QPlayer(Player):
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
            # Player 1 takes action
            actions = self._possible_actions()
            p1_action = self.p1.choose_action(self.board, actions)
            self._update_board(p1_action, self.p1)
            if self._is_finished():
                break

            # Player 2 takes action
            actions = self._possible_actions()
            p2_action = self.p2.choose_action(self.board, actions)
            self._update_board(p2_action, self.p2)
            if self._is_finished():
                break
        winner_name, final_board_state = self._match_summary()
        # Give feedback to the players about the outcome of the match
        self.p1.receive_feedback(winner_name)
        self.p2.receive_feedback(winner_name)
        return winner_name, final_board_state


if __name__ == '__main__':
    """
    Training Documentation
    QPlayer('p1', alpha=0.2, epsilon=0.5, gamma=0.95) on RandomPlayer for 1000 rounds
    QPlayer('p1', alpha=0.2, epsilon=0.3, gamma=0.95) on frozen self for 1000 rounds
    QPlayer('p1', alpha=0.2, epsilon=0.1, gamma=0.95) on RandomPlayer for 3000 rounds
    QPlayer('p1', alpha=0.2, epsilon=0.1, gamma=0.95) on frozen self for 1000 rounds
    """
    p1_ = QPlayer('p1', alpha=0.2, epsilon=0.1, gamma=0.8)
    p1_.load_policy('tictactoe_policy')
    # p2_ = QPlayer('p2', alpha=0.1, epsilon=0.1, gamma=0.8)
    # p2_.load_policy('p1_policy')
    p2_ = RandomPlayer('p2')
    # p2_ = HumanPlayer('Ujil')

    print('training...')
    log_round = list()
    log_total = dict()
    for i in tqdm(range(4000)):
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
    p1_.store_policy('tictactoe_policy')
    print('finished')
