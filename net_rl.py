import os
import random
import numpy as np
from collections import deque
from seed import PACMAN_SEED
import torch
import torch.nn as nn
import torch.optim as optim
import seed  # Импортируем seed как модуль

# This script implements Deep Q-Learning (DQN) for Pacman.
# The agent interacts with the environment, collects experience (state, action, reward, next_state, done),
# and stores it in replay memory. At each step, it selects actions using an epsilon-greedy policy:
# - With probability epsilon, it chooses a random action (exploration).
# - Otherwise, it chooses the action with the highest predicted Q-value (exploitation).
# The neural network (DQNPacmanNet) predicts Q-values for all actions given the current state.
# The agent periodically samples random batches from memory to train the network,
# minimizing the difference between predicted Q-values and target Q-values (Bellman equation).
# The target network is updated every TARGET_UPDATE episodes for stable learning.
# After training, the model is saved to disk.

# Set seeds for reproducibility
torch.manual_seed(PACMAN_SEED)
random.seed(PACMAN_SEED)
np.random.seed(PACMAN_SEED)

# Constants
HIDDEN_SIZE = 128
NUM_ACTIONS = 5  # Stop, North, South, East, West
BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
MEMORY_SIZE = 30000
TARGET_UPDATE = 10
NUM_EPISODES = 3000
MAX_STEPS = 500
MODELS_DIR = "models"

# Action mapping
ACTION_TO_IDX = {
    'Stop': 0,
    'North': 1,
    'South': 2, 
    'East': 3,
    'West': 4
}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

class DQNPacmanNet(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions):
        super(DQNPacmanNet, self).__init__()
        self.input_features = input_shape[0] * input_shape[1]
        self.model = nn.Sequential(
            nn.Linear(self.input_features, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(tuple(args))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        return zip(*batch)
    def __len__(self):
        return len(self.memory)

def select_action(state, policy_net, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        policy_net.eval()  # Ensure eval mode for BatchNorm
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state)
        return q_values.argmax().item()

def optimize_model(policy_net, target_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    policy_net.train()  # Ensure training mode for BatchNorm
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    loss = nn.MSELoss()(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_model(model, input_size, model_path="models/pacman_dqn.pth"):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    # Save both input_size and input_shape for compatibility
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'input_shape': input_size,  # for legacy compatibility
    }
    torch.save(model_info, model_path)
    print(f'Model saved to {model_path}')
    print(f"Saved model keys: {list(model_info.keys())}")

def main():
    import time
    start_time = time.time()
    from pacman_gym_env import PacmanGymEnv
    env = PacmanGymEnv()  # You can pass layout_name if needed

    input_shape = env.observation_space.shape

    if torch.backends.mps.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("Using MPS device for macOS")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using CUDA device" if torch.cuda.is_available() else "Using CPU")
    print(f"Using device: {device}")

    policy_net = DQNPacmanNet(input_shape, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    target_net = DQNPacmanNet(input_shape, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    # Метрики
    episode_rewards = []
    episode_losses = []
    episode_wins = []
    episode_losses_count = []

    for episode in range(NUM_EPISODES):
        # Меняем сид только раз в 500 эпизодов
        if episode % 500 == 0:
            new_seed = PACMAN_SEED + episode // 500
            setattr(seed, 'PACMAN_SEED', new_seed)
            torch.manual_seed(new_seed)
            random.seed(new_seed)
            np.random.seed(new_seed)

        state = env.reset()
        total_reward = 0
        losses = []
        done = False
        win = False
        lose = False

        for t in range(MAX_STEPS):
            action = select_action(state, policy_net, epsilon, device)
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            # Optimize and track loss
            if len(memory) >= BATCH_SIZE:
                policy_net.train()
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
                loss = nn.MSELoss()(q_values, expected_q_values.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                loss = None

            # Проверка на победу/проигрыш (если env возвращает info['win'] или info['lose'])
            # Если нет, используем reward или done
            if done:
                # Попробуйте получить win/lose из info, иначе определяйте по reward
                if 'win' in info:
                    win = info['win']
                    lose = not win
                else:
                    # Примитивная эвристика: если reward большой, значит победа
                    win = reward > 100
                    lose = not win
                break

        avg_loss = np.mean(losses) if losses else 0.0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        episode_wins.append(int(win))
        episode_losses_count.append(int(lose))

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"Total reward: {total_reward:.2f} | "
              f"Avg loss: {avg_loss:.4f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"{'WIN' if win else ('LOSE' if lose else '')} | "
              f"MEMORY_SIZE: {len(memory)}"
              f"SEED: {getattr(seed, 'PACMAN_SEED', 'N/A')}")

        # Каждые 100 эпизодов выводим агрегированные метрики
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss_100 = np.mean([l for l in episode_losses[-100:] if l is not None])
            win_rate = np.sum(episode_wins[-100:]) / 100.0 * 100
            lose_rate = np.sum(episode_losses_count[-100:]) / 100.0 * 100
            print(f"--- Last 100 episodes ---")
            print(f"Avg reward: {avg_reward:.2f} | Avg loss: {avg_loss_100:.4f} | "
                  f"Win rate: {win_rate:.1f}% | Lose rate: {lose_rate:.1f}%")
            print("-------------------------")

    save_model(policy_net, input_shape)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()