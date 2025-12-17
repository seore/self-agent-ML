import random
from collections import deque
import numpy as np
import torch

from .model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  
        self.gamma = 0.9  
        self.memory = deque(maxlen=MAX_MEMORY)

        self.input_size = 11   
        self.hidden_size = 256
        self.output_size = 3  

        self.model = Linear_QNet(self.input_size, self.hidden_size, self.output_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action_one_hot, reward, next_state, done):
        # store index of action (e.g., 0,1,2) instead of full one-hot
        action_idx = np.argmax(action_one_hot)
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action_one_hot, reward, next_state, done):
        action_idx = np.argmax(action_one_hot)
        self.trainer.train_step(state, action_idx, reward, next_state, done)

    def get_action(self, state):
        """
        Epsilon-greedy action selection.
        Returns one-hot vector of length 3: [straight, right, left]
        """
        # Decrease epsilon as games increase (less random over time)
        self.epsilon = max(0, 80 - self.n_games)

        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move
