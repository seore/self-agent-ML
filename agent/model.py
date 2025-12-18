import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str, device: torch.device):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Model file is empty/corrupted: {file_path}")
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.eval()


class DQNTrainer:
    """
    Supports:
      - Target network
      - Double DQN option
    """
    def __init__(self, online_model, target_model, lr, gamma, device, double_dqn: bool):
        self.online = online_model
        self.target = target_model
        self.gamma = gamma
        self.device = device
        self.double_dqn = double_dqn

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def _compute_targets(self, rewards, dones, next_states):
        """
        rewards: (B,)
        dones: (B,) bool
        next_states: (B, state_dim)
        returns target_q: (B,)
        """
        if self.double_dqn:
            # online selects action, target evaluates it
            next_q_online = self.online(next_states)              # (B, A)
            next_actions = torch.argmax(next_q_online, dim=1)     # (B,)
            next_q_target = self.target(next_states)             # (B, A)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            next_q = torch.max(self.target(next_states), dim=1).values  # (B,)

        target_q = rewards + (~dones) * self.gamma * next_q
        return target_q

    def train_batch(self, states, actions, rewards, next_states, dones):
        """
        states: (B, S) float32
        actions: (B,) int64
        rewards: (B,) float32
        next_states: (B, S) float32
        dones: (B,) bool
        """
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        pred_q_all = self.online(states)  # (B, A)
        pred_q = pred_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            target_q = self._compute_targets(rewards, dones, next_states)

        loss = self.criterion(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
