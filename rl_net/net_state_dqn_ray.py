import os
import random
import numpy as np
from collections import deque
from seed import PACMAN_SEED
import torch
import torch.nn as nn
import torch.optim as optim
import seed
import ray

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
NUM_EPISODES = 500
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

class StateValueNet(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(StateValueNet, self).__init__()
        self.input_features = input_shape[0] * input_shape[1]
        self.model = nn.Sequential(
            nn.Linear(self.input_features, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x).squeeze(-1)

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

def optimize_value_net(value_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    value_net.train()
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    states = torch.FloatTensor(np.array(states)).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Value target: reward + gamma * V(next_state) * (1 - done)
    with torch.no_grad():
        next_values = value_net(next_states)
        targets = rewards + (GAMMA * next_values * (1 - dones))
    values = value_net(states)
    loss = nn.MSELoss()(values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

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

def save_value_model(model, input_size, model_path="models/pacman_value.pth"):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'input_shape': input_size,
    }
    torch.save(model_info, model_path)
    print(f'State value model saved to {model_path}')

@ray.remote
class EnvWorker:
    def __init__(self, input_shape, seed_value):
        # Defensive import and dependency check
        try:
            import numpy as np
            import torch
            from rl_net.pacman_gym_env import PacmanGymEnv
        except ImportError as e:
            print(f"[EnvWorker] ImportError: {e}. Make sure all dependencies are installed in the Ray worker environment.")
            raise

        import os
        # Ensure working directory is project directory
        project_dir = "/Users/stanislavgatin/Documents/Curso 24-25/TÉCNICAS Y ALGORITMOS DE BÚSQUEDA/pacman_TAB"
        if os.getcwd() != project_dir:
            print(f"[EnvWorker] Changing working directory from {os.getcwd()} to {project_dir}")
            os.chdir(project_dir)

        self.input_shape = input_shape
        self.seed_value = seed_value
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        # Print environment info for debugging
        print(f"[EnvWorker] CWD: {os.getcwd()}")
        print(f"[EnvWorker] FILES: {os.listdir(os.getcwd())}")
        print(f"[EnvWorker] ENV: {dict(os.environ)}")

        self.env = PacmanGymEnv()
        # Defensive check for layout initialization
        if not hasattr(self.env, "layout") or self.env.layout is None:
            raise RuntimeError(
                "[EnvWorker] PacmanGymEnv.layout is None. "
                "Check that the environment is initialized with a valid layout. "
                "This may be due to missing files or misconfiguration.\n"
                f"Working directory: {os.getcwd()}\n"
                f"Files: {os.listdir(os.getcwd())}\n"
                f"Environment: {dict(os.environ)}"
            )

    def rollout(self, policy_state_dict, epsilon, max_steps):
        # Create a local policy net and load weights
        device = torch.device('cpu')
        policy_net = DQNPacmanNet(self.input_shape, HIDDEN_SIZE, NUM_ACTIONS).to(device)
        policy_net.load_state_dict(policy_state_dict)
        state = self.env.reset()
        episode = []
        total_reward = 0
        for t in range(max_steps):
            if random.random() < epsilon:
                action = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    policy_net.eval()
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(s)
                    action = q_values.argmax().item()
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            if done:
                break
        return episode, total_reward

def main():
    import time
    start_time = time.time()
    from rl_net.pacman_gym_env import PacmanGymEnv
    env = PacmanGymEnv()
    input_shape = env.observation_space.shape

    if torch.backends.mps.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("Using MPS device for macOS")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using CUDA device" if torch.cuda.is_available() else "Using CPU")
    print(f"Using device: {device}")

    pretrained_path = "models/pacman_dqn_v1.0.pth"
    policy_net = DQNPacmanNet(input_shape, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained policy_net from {pretrained_path}")
    else:
        print(f"Pretrained model {pretrained_path} not found. Training from scratch.")

    target_net = DQNPacmanNet(input_shape, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    value_net = StateValueNet(input_shape, HIDDEN_SIZE).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    # Ray initialization
    ray.init(ignore_reinit_error=True)
    num_workers = 4  # You can increase this for more parallelism
    workers = [EnvWorker.remote(input_shape, PACMAN_SEED + i) for i in range(num_workers)]

    episode_rewards = []
    episode_losses = []
    value_losses = []

    for episode in range(NUM_EPISODES):
        iter_start = time.time()
        # Update seed every 500 episodes
        if episode % 500 == 0:
            new_seed = PACMAN_SEED + episode // 500
            setattr(seed, 'PACMAN_SEED', new_seed)
            torch.manual_seed(new_seed)
            random.seed(new_seed)
            np.random.seed(new_seed)

        # Parallel rollout
        policy_state_dict = policy_net.state_dict()
        rollout_ids = [w.rollout.remote(policy_state_dict, epsilon, MAX_STEPS) for w in workers]
        results = ray.get(rollout_ids)
        for episode_data, total_reward in results:
            for transition in episode_data:
                memory.push(*transition)
            episode_rewards.append(total_reward)

        # Training step
        losses = []
        vlosses = []
        for _ in range(len(results)):  # One update per worker
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
                vloss = optimize_value_net(value_net, memory, value_optimizer, device)
                if vloss is not None:
                    vlosses.append(vloss)

        avg_loss = np.mean(losses) if losses else 0.0
        avg_vloss = np.mean(vlosses) if vlosses else 0.0
        episode_losses.append(avg_loss)
        value_losses.append(avg_vloss)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"Avg reward: {np.mean(episode_rewards[-num_workers:]):.2f} | "
              f"Avg loss: {avg_loss:.4f} | "
              f"Value loss: {avg_vloss:.4f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"MEMORY_SIZE: {len(memory)}"
              f"SEED: {getattr(seed, 'PACMAN_SEED', 'N/A')} | "
              f"Step time: {time.time() - iter_start:.2f}s")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100 * num_workers:])
            avg_loss_100 = np.mean([l for l in episode_losses[-100:] if l is not None])
            avg_vloss_100 = np.mean([l for l in value_losses[-100:] if l is not None])
            print(f"--- Last 100 episodes ---")
            print(f"Avg reward: {avg_reward:.2f} | Avg loss: {avg_loss_100:.4f} | "
                  f"Avg value loss: {avg_vloss_100:.4f}")
            print("-------------------------")

    save_model(policy_net, input_shape, model_path="models/pacman_dqn_v1.0_finetuned.pth")
    save_value_model(value_net, input_shape, model_path="models/pacman_value_v1.0_finetuned.pth")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ray.shutdown()

if __name__ == "__main__":
    main()
