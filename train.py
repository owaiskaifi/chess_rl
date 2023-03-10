import numpy as np
import chess
import random
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.4, epsilon=1, discount=0.9):
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # exploration rate
        self.discount = discount # discount factor
        self.q_values = {}

    def get_q_value(self, state, action):
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        return self.q_values[(state, action)]

    def choose_action(self, state, legal_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in legal_actions]
            max_q_value = max(q_values)
            best_actions = [action for action, q_value in zip(legal_actions, q_values) if q_value == max_q_value]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        q_value = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, next_action) for next_action in chess.Board(next_state).legal_moves]
        if next_q_values:
            max_next_q_value = max(next_q_values)
        else:
            max_next_q_value = 0.0
        new_q_value = q_value + self.alpha * (reward + self.discount * max_next_q_value - q_value)
        self.q_values[(state, action)] = new_q_value

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_values, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_values = pickle.load(f)

# class ChessGame:
#     def __init__(self, white_agent, black_agent):
#         self.board = chess.Board()
#         self.white_agent = white_agent
#         self.black_agent = black_agent

#     def play(self):
#         while not self.board.is_game_over():
#             print(self.board)
#             if self.board.turn == chess.WHITE:
#                 state = self.board.fen()
#                 legal_actions = [move.uci() for move in self.board.legal_moves]
#                 action = self.white_agent.choose_action(state, legal_actions)
#                 move = chess.Move.from_uci(action)
#                 self.board.push(move)
#                 print("White agent plays: " + str(move))
#                 reward = self.get_reward()
#                 next_state = self.board.fen()
#                 self.white_agent.update(state, action, reward, next_state)
#             else:
#                 state = self.board.fen()
#                 legal_actions = [move.uci() for move in self.board.legal_moves]
#                 action = self.black_agent.choose_action(state, legal_actions)
#                 move = chess.Move.from_uci(action)
#                 self.board.push(move)
#                 print("Black agent plays: " + str(move))
#                 reward = self.get_reward()
#                 next_state = self.board.fen()
#                 self.black_agent.update(state, action, reward, next_state)

#         print(self.board.result())
#         self.white_agent.save_model('white_agent_model.pkl')
#         self.black_agent.save_model('black_agent_model.pkl')
class ChessGame:
    def __init__(self, white_agent, black_agent):
        self.board = chess.Board()
        self.white_agent = white_agent
        self.black_agent = black_agent

    def play(self, num_rounds):
        for i in range(num_rounds):
            print(f"Round {i+1}/{num_rounds}")

            self.board.reset()

            while not self.board.is_game_over():
                if self.board.turn == chess.WHITE:
                    current_agent = self.white_agent
                else:
                    current_agent = self.black_agent

                legal_moves = list(self.board.legal_moves)
                current_state = self.board.fen()
                action = current_agent.choose_action(current_state, legal_moves)
                self.board.push(action)
                next_state = self.board.fen()

                if self.board.is_checkmate():
                    if self.board.turn == chess.WHITE:
                        reward = -1.0
                        winner = "black"
                    else:
                        reward = 1.0
                        winner = "white"
                elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
                    reward = 0.0
                    winner = "draw"
                else:
                    reward = None
                    winner = None

                if reward is not None:
                    current_agent.update(current_state, action, reward, next_state)

            self.white_agent.save_model(f"white_agent_round.pkl")
            self.black_agent.save_model(f"black_agent_round.pkl")

            print(f"Winner: {winner}")




    def get_reward(self):
        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
       
white_agent = QLearningAgent()
black_agent = QLearningAgent()

game = ChessGame(white_agent, black_agent)
game.play(50)