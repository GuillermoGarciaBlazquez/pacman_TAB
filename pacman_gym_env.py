import gym
import numpy as np
from gym import spaces
from pacman import GameState, ClassicGameRules
import layout

class PacmanGymEnv(gym.Env):
    """
    Minimal OpenAI Gym wrapper for Pacman.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, layout_name="mediumClassic"):
        super().__init__()
        self.layout = layout.getLayout(layout_name)
        self.rules = ClassicGameRules()
        self.game = None
        self.state = None
        self.done = False
        self.prev_score = 0  # Для хранения предыдущего счёта

        # Example: 5 actions (Stop, North, South, East, West)
        self.action_space = spaces.Discrete(5)
        # Observation: map matrix (width x height)
        width, height = self.layout.width, self.layout.height
        self.observation_space = spaces.Box(low=0, high=6, shape=(width, height), dtype=np.float32)

    def reset(self):
        # Start a new game
        self.state = GameState()
        self.state.initialize(self.layout)
        self.done = False
        self.prev_score = self.state.getScore()  # Сброс предыдущего счёта
        return self._get_obs()

    def step(self, action):
        # Map action index to string
        idx_to_action = {0: "Stop", 1: "North", 2: "South", 3: "East", 4: "West"}
        action_str = idx_to_action[action]
        reward = 0
        info = {}

        if self.done:
            return self._get_obs(), 0, True, info

        try:
            self.state = self.state.generateSuccessor(0, action_str)
            current_score = self.state.getScore()
            reward = current_score - self.prev_score
            self.prev_score = current_score
            self.done = self.state.isWin() or self.state.isLose()
        except Exception:
            self.done = True

        return self._get_obs(), reward, self.done, info

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def _get_obs(self):
        # Convert state to numeric matrix (similar to NeuralAgent.state_to_matrix)
        walls = self.state.getWalls()
        width, height = walls.width, walls.height
        numeric_map = np.zeros((width, height), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        food = self.state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        for x, y in self.state.getCapsules():
            numeric_map[x][y] = 3
        for ghost_state in self.state.getGhostStates():
            gx, gy = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            if ghost_state.scaredTimer > 0:
                numeric_map[gx][gy] = 6
            else:
                numeric_map[gx][gy] = 4
        px, py = self.state.getPacmanPosition()
        numeric_map[int(px)][int(py)] = 5
        numeric_map = numeric_map / 6.0
        return numeric_map

# Example usage:
# env = PacmanGymEnv()
# obs = env.reset()
# obs, reward, done, info = env.step(1)  # Take action "North"
