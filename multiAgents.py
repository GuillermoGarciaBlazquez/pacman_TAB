# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
from seed import PACMAN_SEED
random.seed(PACMAN_SEED)  # For reproducibility
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '4'):
        super().__init__(evalFn, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the alpha-beta action using self.depth and self.evaluationFunction
        """
        
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: Check if the game is over or if we've reached the maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximizer) is agentIndex 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            # Ghosts (minimizer) are agentIndex 1 or higher
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize max value
            v = float('-inf')
            # Get Pacman's legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(1, depth, successor, alpha, beta))  # Ghosts start at index 1
                if v > beta:
                    return v  # Prune the remaining branches
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize min value
            v = float('inf')
            # Get the current agent's legal actions (ghosts)
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Get the next agent's index and check if we need to increase depth
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Go back to Pacman
                depth += 1  # Increase the depth since we've gone through all agents

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:
                    return v  # Prune the remaining branches
                beta = min(beta, v)
            return v

        # Pacman (agentIndex 0) will choose the action with the best alpha-beta score
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    A Pacman agent that uses a neural network to make decisions
    based on the evaluation of the game state.
    """
    def __init__(self, model_path="models/pacman_dqn_v2.2.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapping from indices to actions
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # For evaluating alternatives
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Move counter
        self.move_count = 0
        
        print(f"NeuralAgent initialized, using device: {self.device}")

    def load_model(self, model_path):
        """Loads the model from the saved file"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model not found at {model_path}")
                return False
                
            # Load the model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Create and load the model
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Warning: Model state_dict mismatch or error: {e}")
                return False
            self.model.eval()  # Always set to eval mode for inference
            print(f"Model loaded successfully from {model_path}")
            print(f"Input size: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def state_to_matrix(self, state):
        """Converts the game state into a normalized numeric matrix"""
        # Get board dimensions
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Create a numeric matrix
        # 0: wall, 1: empty space, 2: food, 3: capsule, 4: ghost, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Set empty spaces (everything that's not a wall starts as empty)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Add food
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Add capsules
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Add ghosts
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # If the ghost is scared, mark it differently
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Scared ghost
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Normal ghost
        
        # Add Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalize
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        An evaluation function based on the neural network and additional heuristics.
        """
        if self.model is None:
            return 0  # If no model, return 0
        
        # Convert to matrix
        state_matrix = self.state_to_matrix(state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Always set model to eval mode before inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Get legal actions
        legal_actions = state.getLegalActions()
        
        # Apply additional heuristics, similar to betterEvaluationFunction
        score = state.getScore()
        
        # Improve evaluation with domain knowledge
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distance to the nearest food
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximity to ghosts
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # If the ghost is scared, approach it
                score += 50 / (ghost_distance + 1)
            else:
                # If not scared, avoid it
                if ghost_distance <= 2:
                    score -= 200  # Large penalty for being too close
        
        # Combine the neural network score with the heuristic
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Returns the best action based on the neural network evaluation
        and additional heuristics.
        """
        self.move_count += 1
        
        # If no model, make a random move
        if self.model is None:
            print("ERROR: Model not loaded. Making random move.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Get legal actions
        legal_actions = state.getLegalActions()
        
        # Direct evaluation with the neural network
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Always set model to eval mode before inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Map model indices to game actions
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Sort by probability (descending)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: with a decreasing probability, choose randomly
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Decreases over time
        if random.random() < exploration_rate:
            # Exclude STOP if possible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Alternative evaluation: generate successors and evaluate each
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combine heuristic evaluation with the network prediction
            combined_score = eval_score + neural_score
            
            # Penalize STOP unless it's the only option
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Sort by combined score
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best action
        return successors[0][0]

# Define a function to create the agent
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Factory function to create a neural agent.
    Useful for integration with pacman.py structure.
    """
    return NeuralAgent(model_path)

class AlphaBetaNeuralAgent(Agent):
    """
    Pacman agent using alpha-beta pruning and a value network to evaluate states.
    """
    def __init__(self, model_path="models/pacman_value_v1.0_finetuned.pth", depth=3):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth = depth
        self.load_model(model_path)
        print(f"AlphaBetaNeuralAgent initialized, device: {self.device}, depth: {self.depth}")

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Value model not found at {model_path}")
                return False
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            # ValueNet: input_shape, hidden_size
            from rl_net.net_state_dqn import StateValueNet
            self.model = StateValueNet(self.input_size, 128).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Value model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading value model: {e}")
            return False

    def state_to_matrix(self, state):
        """Converts the game state into a normalized numeric matrix"""
        # Get board dimensions
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Create a numeric matrix
        # 0: wall, 1: empty space, 2: food, 3: capsule, 4: ghost, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Set empty spaces (everything that's not a wall starts as empty)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Add food
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Add capsules
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Add ghosts
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # If the ghost is scared, mark it differently
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Scared ghost
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Normal ghost
        
        # Add Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalize
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def value_evaluate(self, state):
        if self.model is None:
            return 0
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.model(state_tensor).cpu().item()
        return value

    def getAction(self, state):
        """
        Alpha-beta pruning with a value network for state evaluation.
        """
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.value_evaluate(gameState)
            num_agents = gameState.getNumAgents()
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.value_evaluate(gameState)
            if agentIndex == 0:  # Pacman (max)
                v = float('-inf')
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = max(v, alphabeta(1, depth, successor, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:  # Ghosts (min)
                v = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == num_agents:
                    nextAgent = 0
                    nextDepth += 1
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)
        return bestAction

# Define a function to create the agent
def createAlphaBetaNeuralAgent(model_path="models/pacman_value.pth", depth=3):
    """
    Factory to create AlphaBetaNeuralAgent.
    """
    return AlphaBetaNeuralAgent(model_path, depth)
