import os
import random
import numpy as np
import torch

from .model import Linear_QNet, DQNTrainer
from collections import deque


class Agent:
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        lr: float,
        gamma: float,
        max_memory: int,
        batch_size: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_games: int,
        target_update_steps: int,
        double_dqn: bool,
        device: torch.device,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.device = device

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_games = max(1, epsilon_decay_games)

        self.target_update_steps = target_update_steps
        self.double_dqn = double_dqn

        self.n_games = 0
        self.steps = 0
        self.memory = deque(maxlen=max_memory)

        self.online = Linear_QNet(input_size, hidden_size, output_size).to(self.device)
        self.target = Linear_QNet(input_size, hidden_size, output_size).to(self.device)
        self.sync_target(hard=True)

        self.trainer = DQNTrainer(
            online_model=self.online,
            target_model=self.target,
            lr=lr,
            gamma=gamma,
            device=self.device,
            double_dqn=double_dqn,
        )

    def epsilon(self):
        # linear decay over games
        t = min(self.n_games / self.epsilon_decay_games, 1.0)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def sync_target(self, hard=False):
        self.target.load_state_dict(self.online.state_dict())

    def remember(self, state, action_one_hot, reward, next_state, done):
        action_idx = int(np.argmax(action_one_hot))
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state, greedy=False):
        if (not greedy) and (random.random() < self.epsilon()):
            a = random.randint(0, self.output_size - 1)
            one_hot = [0] * self.output_size
            one_hot[a] = 1
            return one_hot

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(state_t)[0]
            a = int(torch.argmax(q).item())
        one_hot = [0] * self.output_size
        one_hot[a] = 1
        return one_hot

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        loss = self.trainer.train_batch(
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

        self.steps += 1
        if self.steps % self.target_update_steps == 0:
            self.sync_target(hard=True)

        return loss

    def save(self, path: str):
        self.online.save(path)

    def load(self, path: str):
        self.online.load(path, self.device)
        self.sync_target(hard=True)