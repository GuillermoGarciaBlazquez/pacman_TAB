import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ... ValueNet как выше ...

value_net = ValueNet(input_shape, hidden_size).to(device)
optimizer = optim.Adam(value_net.parameters(), lr=0.001)
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # ...выберите действие (например, случайно или policy)...
        next_state, reward, done, _ = env.step(action)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            target = reward + gamma * value_net(next_state_tensor) * (1 - done)
        value = value_net(state_tensor)
        loss = (value - target).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state
