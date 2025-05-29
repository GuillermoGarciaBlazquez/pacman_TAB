import os
import random
import numpy as np
from collections import deque
from seed import PACMAN_SEED
import torch
import torch.nn as nn
import torch.optim as optim
import seed  # Import seed as a module
import sys
import csv
import matplotlib.pyplot as plt

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
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 15000  # Number of episodes to decay epsilon from EPS_START to EPS_END
TAU = 0.0005  # Soft update parameter
MEMORY_SIZE = 50000
TARGET_UPDATE = 1000
NUM_EPISODES = 30000
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
            # nn.BatchNorm1d(hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            # nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
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
    # Always ensure eval mode for deterministic action selection (BatchNorm & Dropout OFF)
    policy_net.eval()
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
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
    q_values = torch.clamp(q_values, -50, 50)
    # --- Double DQN target calculation ---
    with torch.no_grad():
        # 1. Select actions with maximum Q from policy_net
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        # 2. Get Q of these actions from target_net
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
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

# --- Metrics/Logging/Plotting Setup ---
MODEL_VERSION = "v2.2"
MODEL_NAME = f"pacman_dqn_{MODEL_VERSION}"
LOGS_DIR = "logs"
METRICS_DIR = "metrics"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"{MODEL_NAME}.log")
CSV_FILE = os.path.join(METRICS_DIR, f"{MODEL_NAME}_metrics.csv")
PLOT_FILE = os.path.join(METRICS_DIR, f"{MODEL_NAME}_metrics.png")

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(LOG_FILE)

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
    # Always set target_net to eval mode (no Dropout/BatchNorm randomness)
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    # Metrics
    episode_rewards = []
    episode_losses = []
    episode_wins = []
    episode_losses_count = []
    episode_lengths = []
    episode_max_q = []
    episode_min_q = []
    episode_mean_q = []
    episode_loss_std = []
    episode_reward_std = []

    # Prepare CSV for metrics
    with open(CSV_FILE, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "episode", "reward", "avg_loss", "loss_std", "reward_std", "win", "lose", "length",
            "max_q", "min_q", "mean_q", "epsilon"
        ])

    best_reward = float('-inf')
    best_win_rate = float('-inf')
    best_reward_model_path = f"models/{MODEL_NAME}_best_reward.pth"
    best_winrate_model_path = f"models/{MODEL_NAME}_best_winrate.pth"
    best_reward_episode = 0
    best_winrate_episode = 0

    for episode in range(NUM_EPISODES):
        # Change seed only once every 500 episodes
        # if episode % 500 == 0:
        #     new_seed = PACMAN_SEED + episode // 500
        #     setattr(seed, 'PACMAN_SEED', new_seed)
        #     torch.manual_seed(new_seed)
        #     random.seed(new_seed)
        #     np.random.seed(new_seed)

        state = env.reset()
        total_reward = 0
        losses = []
        q_values_list = []
        done = False
        win = False
        lose = False

        for t in range(MAX_STEPS):
            action = select_action(state, policy_net, epsilon, device)
            # --- Metrics: Q-values ---
            # Always ensure eval mode before Q-value logging
            policy_net.eval()
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(s).cpu().numpy().flatten()
                q_values_list.append(q_values)
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
                # --- Double DQN target calculation ---
                with torch.no_grad():
                    # 1. Select actions with maximum Q from policy_net
                    next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    # 2. Get Q of these actions from target_net
                    next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
                loss = nn.MSELoss()(q_values, expected_q_values.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                loss = None

            # Check for win/lose via info['win'] and done
            if done:
                win = bool(info.get('win', False))
                lose = not win
                break

        avg_loss = np.mean(losses) if losses else 0.0
        std_loss = np.std(losses) if losses else 0.0
        std_reward = np.std([total_reward])  # For single episode, will be 0, but for batch can be used
        q_values_arr = np.array(q_values_list).reshape(-1, NUM_ACTIONS) if q_values_list else np.zeros((1, NUM_ACTIONS))
        max_q = np.max(q_values_arr)
        min_q = np.min(q_values_arr)
        mean_q = np.mean(q_values_arr)

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        episode_wins.append(int(win))
        episode_losses_count.append(int(lose))
        episode_lengths.append(t + 1)
        episode_max_q.append(max_q)
        episode_min_q.append(min_q)
        episode_mean_q.append(mean_q)
        episode_loss_std.append(std_loss)
        episode_reward_std.append(std_reward)

        # Save metrics to CSV
        with open(CSV_FILE, "a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                episode+1, total_reward, avg_loss, std_loss, std_reward, int(win), int(lose), t+1,
                max_q, min_q, mean_q, epsilon
            ])

        # Linear epsilon decay
        epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_STEPS))

        # --- Soft update of the target network ---
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (1 - TAU) * target_net_state_dict[key] + TAU * policy_net_state_dict[key]
        target_net.load_state_dict(target_net_state_dict)

        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"Total reward: {total_reward:.2f} | "
              f"Avg loss: {avg_loss:.4f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"{'WIN' if win else ('LOSE' if lose else '')} | "
              f"MEMORY_SIZE: {len(memory)} | "
              f"SEED: {getattr(seed, 'PACMAN_SEED', 'N/A')}")
               # Every 100 episodes, print aggregated metrics
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss_100 = np.mean([l for l in episode_losses[-100:] if l is not None])
            win_rate = np.sum(episode_wins[-100:]) / 100.0 * 100
            lose_rate = np.sum(episode_losses_count[-100:]) / 100.0 * 100
            print(f"--- Last 100 episodes ---")
            print(f"Avg reward: {avg_reward:.2f} | Avg loss: {avg_loss_100:.4f} | "
                  f"Win rate: {win_rate:.1f}% | Lose rate: {lose_rate:.1f}%")
            print("-------------------------")

            # Save best by reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_reward_episode = episode + 1
                save_model(policy_net, input_shape, model_path=best_reward_model_path)
                print(f"Best avg reward model saved at episode {best_reward_episode} with avg reward {best_reward:.2f}")

            # Save best by win rate
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_winrate_episode = episode + 1
                save_model(policy_net, input_shape, model_path=best_winrate_model_path)
                print(f"Best win rate model saved at episode {best_winrate_episode} with win rate {best_win_rate:.2f}%")


    # Save last model as before
    save_model(policy_net, input_shape, model_path=f"models/{MODEL_NAME}.pth")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # Plot metrics
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(episode_losses, label="Avg Loss")
    plt.plot(episode_loss_std, label="Loss Std", alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(episode_max_q, label="Max Q")
    plt.plot(episode_min_q, label="Min Q")
    plt.plot(episode_mean_q, label="Mean Q")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("Q-values")
    plt.legend()

    plt.subplot(2, 2, 4)
    win_rate_curve = np.convolve(episode_wins, np.ones(100)/100, mode='valid')
    plt.plot(win_rate_curve, label="Win Rate (window=100)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Saved metrics plot to {PLOT_FILE}")

if __name__ == "__main__":
    main()