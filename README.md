Python Chess RL Game

This is a chess game implemented in Python using the python-chess library and enhanced with reinforcement learning (RL) techniques. The game can be played against an AI agent that learns from its mistakes and improves its strategy over time.
Requirements

    Python 3.x
    python-chess library


Installation

    Clone this repository: git clone https://github.com/yourusername/Python-Chess-RL-Game.git
    Run the training session: python train.py
    Run the test session : python test.py

Usage




To let two AI agents play against each other, choose the Train.py file. The two agents will take turns making their moves until the game is over and saves a model. You can specify the number of rounds to play by entering a positive integer. At the end of each round, the learned policy of the winning agent will be saved to a model file with the name you specified.
Loading a saved model

You can load a saved model file. The AI agent will use the learned policy from the model to make its moves.

TODO:
  choose to play against the AI agent
