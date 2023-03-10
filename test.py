import chess
import pickle


import random

class RLAgent:
    def __init__(self, alpha=0.4, epsilon=1, discount=0.9):
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = discount
        self.prev_state = None
        self.prev_action = None

    def get_Q(self, state, action):
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0
        return self.Q[state][action]

    def choose_action(self, state, legal_moves):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(legal_moves)
        else:
            q_values = [self.get_Q(state, a) for a in legal_moves]
            max_q = max(q_values)
            if q_values.count(max_q) > 1:
                best_moves = [m for m in legal_moves if self.get_Q(state, m) == max_q]
                action = random.choice(best_moves)
            else:
                index = q_values.index(max_q)
                action = legal_moves[index]

        self.prev_state = state
        self.prev_action = action
        return action

    def update_Q(self, reward, new_state):
        if self.prev_state is not None:
            old_q = self.get_Q(self.prev_state, self.prev_action)
            max_q = max([self.get_Q(new_state, a) for a in self.get_legal_moves(new_state)])
            self.Q[self.prev_state][self.prev_action] += self.alpha * (reward + self.gamma * max_q - old_q)

    def get_legal_moves(self, state):
        board = chess.Board(state)
        return list(board.legal_moves)


class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def play_game(self, white_model_file, black_model_file):
        # Load models
        with open(white_model_file, "rb") as f:
            white_model = pickle.load(f)
        with open(black_model_file, "rb") as f:
            black_model = pickle.load(f)

        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                agent=RLAgent()
                agent.Q=white_model
                current_agent = agent
            else:
                agent=RLAgent()
                agent.Q=black_model
                current_agent = agent

            legal_moves = list(self.board.legal_moves)
            current_state = self.board.fen()

            action = current_agent.choose_action(current_state, legal_moves)
            self.board.push(action)

            print(self.board)
            print("")

        if self.board.result() == "1-0":
            print("White wins!")
        elif self.board.result() == "0-1":
            print("Black wins!")
        else:
            print("Draw!")
game = ChessGame( )
game.play_game('white_agent.pkl',  'black_agent.pkl'               )