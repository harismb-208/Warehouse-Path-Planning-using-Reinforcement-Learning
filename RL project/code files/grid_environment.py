# --- grid_environment.py ---
import numpy as np
import config

class GridEnvironment:
    """
    Represents the warehouse grid environment for the robot.

    This class manages the grid layout, states, actions, rewards, and transition dynamics
    as defined in the config.py file.
    """
    def __init__(self):
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.start_pos = config.START_POS
        self.goal_pos = config.GOAL_POS
        self.obstacles = set(config.OBSTACLES)
        
        self.actions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        self.states = self._get_states()
        
        # Validate that start and goal are not obstacles
        if self.start_pos in self.obstacles:
            raise ValueError("Start position cannot be an obstacle.")
        if self.goal_pos in self.obstacles:
            raise ValueError("Goal position cannot be an obstacle.")

    def _get_states(self):
        """Returns a list of all valid (non-obstacle) states (row, col)."""
        states = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.obstacles:
                    states.append((r, c))
        return states

    def get_reward(self, state):
        """Returns the reward for a given state."""
        if state == self.goal_pos:
            return config.REWARD_GOAL
        return config.REWARD_STEP

    def get_transition_probabilities(self, state, action):
        """
        Calculates the transition probabilities for a given state and action.

        Args:
            state (tuple): The current state (row, col).
            action (str): The intended action ('UP', 'DOWN', 'LEFT', 'RIGHT').

        Returns:
            list: A list of tuples, where each tuple contains
                  (probability, next_state, reward).
        """
        if state == self.goal_pos:
            # Terminal state: stay in place with 0 reward
            return [(1.0, state, 0)]

        if config.TRANSITION_MODEL_TYPE == 'deterministic':
            return self._get_deterministic_transitions(state, action)
        elif config.TRANSITION_MODEL_TYPE == 'stochastic':
            return self._get_stochastic_transitions(state, action)
        else:
            raise ValueError("Invalid transition model type in config.")

    def _calculate_next_state(self, state, move):
        """Calculates the resulting state after a move, handling walls and obstacles."""
        next_r = state[0] + move[0]
        next_c = state[1] + move[1]
        next_state = (next_r, next_c)

        # Check for boundary collisions
        if not (0 <= next_r < self.height and 0 <= next_c < self.width):
            return state, config.REWARD_OBSTACLE
            
        # Check for obstacle collisions
        if next_state in self.obstacles:
            return state, config.REWARD_OBSTACLE
        
        return next_state, self.get_reward(next_state)

    def _get_deterministic_transitions(self, state, action):
        """Handles deterministic transitions."""
        move = self.actions[action]
        next_state, reward = self._calculate_next_state(state, move)
        # In deterministic model, reward is based on outcome, but let's
        # simplify and say the step cost is always there unless it's a wall bump
        if reward != config.REWARD_OBSTACLE:
             reward = config.REWARD_STEP
        if next_state == self.goal_pos:
             reward = config.REWARD_GOAL
        
        return [(1.0, next_state, reward)]

    def _get_stochastic_transitions(self, state, action):
        """Handles stochastic transitions based on config probabilities."""
        transitions = []
        
        # Define actions relative to the intended action
        if action == 'UP':
            action_options = {'forward': 'UP', 'left': 'LEFT', 'right': 'RIGHT'}
        elif action == 'DOWN':
            action_options = {'forward': 'DOWN', 'left': 'RIGHT', 'right': 'LEFT'}
        elif action == 'LEFT':
            action_options = {'forward': 'LEFT', 'left': 'DOWN', 'right': 'UP'}
        else: # RIGHT
            action_options = {'forward': 'RIGHT', 'left': 'UP', 'right': 'DOWN'}
            
        action_probs = {
            'forward': config.PROB_FORWARD,
            'left': config.PROB_LEFT,
            'right': config.PROB_RIGHT
        }

        for move_type, prob in action_probs.items():
            if prob > 0:
                actual_action = action_options[move_type]
                move = self.actions[actual_action]
                next_state, move_reward = self._calculate_next_state(state, move)
                
                # Check if this outcome already exists to merge probabilities
                found = False
                for i, (p, s, r) in enumerate(transitions):
                    if s == next_state:
                        # Update reward if the new path to this state is better (less likely but possible)
                        # A simple merge: add probabilities, average rewards (weighted)
                        new_prob = p + prob
                        new_reward = (r * p + move_reward * prob) / new_prob
                        transitions[i] = (new_prob, s, new_reward)
                        found = True
                        break
                if not found:
                    transitions.append((prob, next_state, move_reward))

        return transitions
