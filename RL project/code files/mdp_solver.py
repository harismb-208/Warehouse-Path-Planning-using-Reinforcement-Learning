# --- mdp_solver.py ---
import numpy as np
import time
from tqdm import tqdm
import config

class MDPSolver:
    """
    Solves the MDP defined by the GridEnvironment using Value and Policy Iteration.
    """
    def __init__(self, environment):
        self.env = environment
        self.gamma = config.DISCOUNT_FACTOR
        self.theta = config.CONVERGENCE_THRESHOLD

    def _initialize_values(self):
        """Initializes the value function for all states to zero."""
        return {state: 0.0 for state in self.env.states}

    def _initialize_policy(self):
        """Initializes a random policy for all states."""
        return {state: np.random.choice(list(self.env.actions.keys())) for state in self.env.states}

    def value_iteration(self):
        """
        Performs Value Iteration to find the optimal value function and policy.
        
        Returns:
            tuple: (optimal policy, optimal value function, number of iterations, execution time)
        """
        print("--- Starting Value Iteration ---")
        start_time = time.time()
        V = self._initialize_values()
        iterations = 0
        
        while True:
            iterations += 1
            delta = 0
            V_new = V.copy()
            
            pbar = tqdm(self.env.states, desc=f"Value Iteration {iterations}")
            for state in pbar:
                if state == self.env.goal_pos:
                    continue

                action_values = []
                for action in self.env.actions:
                    q_value = 0
                    for prob, next_state, reward in self.env.get_transition_probabilities(state, action):
                        q_value += prob * (reward + self.gamma * V[next_state])
                    action_values.append(q_value)
                
                best_value = max(action_values)
                delta = max(delta, abs(best_value - V[state]))
                V_new[state] = best_value
            
            V = V_new
            pbar.set_postfix({"Delta": f"{delta:.6f}"})
            if delta < self.theta:
                break
        
        # Extract optimal policy
        policy = self._extract_policy(V)
        end_time = time.time()
        
        print(f"Value Iteration converged in {iterations} iterations.")
        return policy, V, iterations, end_time - start_time

    def policy_iteration(self):
        """
        Performs Policy Iteration to find the optimal policy and value function.
        
        Returns:
            tuple: (optimal policy, optimal value function, number of iterations, execution time)
        """
        print("\n--- Starting Policy Iteration ---")
        start_time = time.time()
        policy = self._initialize_policy()
        V = self._initialize_values()
        iterations = 0

        while True:
            iterations += 1
            print(f"Policy Iteration: Cycle {iterations}")
            
            # 1. Policy Evaluation
            V = self._policy_evaluation(policy, V)

            # 2. Policy Improvement
            policy_stable = True
            new_policy = policy.copy()
            
            pbar_improve = tqdm(self.env.states, desc="Policy Improvement")
            for state in pbar_improve:
                if state == self.env.goal_pos:
                    continue
                
                old_action = policy[state]
                
                action_values = {}
                for action in self.env.actions:
                    q_value = 0
                    for prob, next_state, reward in self.env.get_transition_probabilities(state, action):
                        q_value += prob * (reward + self.gamma * V[next_state])
                    action_values[action] = q_value
                
                best_action = max(action_values, key=action_values.get)
                new_policy[state] = best_action
                
                if old_action != best_action:
                    policy_stable = False
            
            policy = new_policy
            if policy_stable:
                break
        
        end_time = time.time()
        print(f"Policy Iteration converged in {iterations} cycles.")
        return policy, V, iterations, end_time - start_time

    def _policy_evaluation(self, policy, V_init):
        """
        Evaluates a given policy by calculating its value function.
        
        Args:
            policy (dict): The policy to evaluate.
            V_init (dict): Initial value function, can be from previous iteration.

        Returns:
            dict: The calculated value function for the policy.
        """
        V = V_init.copy()
        eval_iter = 0
        while True:
            eval_iter += 1
            delta = 0
            V_new = V.copy()
            pbar_eval = tqdm(self.env.states, desc=f"Policy Evaluation {eval_iter}", leave=False)
            for state in pbar_eval:
                if state == self.env.goal_pos:
                    continue
                
                action = policy[state]
                new_value = 0
                for prob, next_state, reward in self.env.get_transition_probabilities(state, action):
                    new_value += prob * (reward + self.gamma * V[next_state])
                
                delta = max(delta, abs(new_value - V[state]))
                V_new[state] = new_value
            
            V = V_new
            pbar_eval.set_postfix({"Delta": f"{delta:.6f}"})
            if delta < self.theta:
                break
        return V

    def _extract_policy(self, V):
        """Extracts a greedy policy from a value function."""
        policy = {}
        for state in self.env.states:
            if state == self.env.goal_pos:
                policy[state] = 'GOAL'
                continue

            action_values = {}
            for action in self.env.actions:
                q_value = 0
                for prob, next_state, reward in self.env.get_transition_probabilities(state, action):
                    q_value += prob * (reward + self.gamma * V[next_state])
                action_values[action] = q_value
            
            best_action = max(action_values, key=action_values.get)
            policy[state] = best_action
        return policy
