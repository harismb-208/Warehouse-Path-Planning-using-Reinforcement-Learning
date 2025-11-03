# --- evaluation.py ---
import numpy as np
import time
from tqdm import tqdm
import config

def simulate_run(env, policy):
    """
    Simulates a single run of the robot following a policy in the environment.

    Args:
        env (GridEnvironment): The environment instance.
        policy (dict): The policy to follow.

    Returns:
        tuple: (success, path_length) where success is a boolean.
    """
    state = env.start_pos
    path_length = 0
    for _ in range(config.MAX_PATH_LENGTH):
        if state == env.goal_pos:
            return True, path_length
        
        action = policy.get(state)
        if not action or action == 'GOAL':
            # No policy for this state, or it's a terminal state without being the goal
            return False, path_length 

        # Get possible transitions for the action from policy
        transitions = env.get_transition_probabilities(state, action)
        
        # Sample the next state based on transition probabilities
        probs = [t[0] for t in transitions]
        next_states = [t[1] for t in transitions]
        
        # Use numpy for sampling
        idx = np.random.choice(len(next_states), p=probs)
        state = next_states[idx]
        
        path_length += 1
    
    return False, path_length # Failed to reach goal within max steps

def evaluate_policy(env, policy):
    """
    Evaluates a policy by running multiple simulations.

    Args:
        env (GridEnvironment): The environment instance.
        policy (dict): The policy to evaluate.

    Returns:
        tuple: (success_rate, average_path_length)
    """
    successes = 0
    total_path_length = 0
    
    print(f"\nEvaluating policy over {config.NUM_SIMULATIONS} simulations...")
    for _ in tqdm(range(config.NUM_SIMULATIONS)):
        success, path_length = simulate_run(env, policy)
        if success:
            successes += 1
            total_path_length += path_length
            
    success_rate = (successes / config.NUM_SIMULATIONS) * 100
    avg_path_length = total_path_length / successes if successes > 0 else float('inf')
    
    return success_rate, avg_path_length

def compare_algorithms(vi_results, pi_results, env):
    """
    Compares the results of Value Iteration and Policy Iteration and prints a summary.
    """
    policy_vi, _, iters_vi, time_vi = vi_results
    policy_pi, _, iters_pi, time_pi = pi_results

    # Evaluate policies
    success_rate_vi, avg_len_vi = evaluate_policy(env, policy_vi)
    success_rate_pi, avg_len_pi = evaluate_policy(env, policy_pi)

    print("\n" + "="*50)
    print("      MDP ALGORITHM COMPARISON REPORT")
    print("="*50)
    print(f"Environment: {env.width}x{env.height} Grid")
    print(f"Transition Model: {config.TRANSITION_MODEL_TYPE}")
    print(f"Discount Factor (Gamma): {config.DISCOUNT_FACTOR}")
    print(f"Convergence Threshold (Theta): {config.CONVERGENCE_THRESHOLD}")
    print("-"*50)
    print(f"Metric{'':<15} | Value Iteration{'':<5} | Policy Iteration")
    print("-"*50)
    print(f"Execution Time (s){'':<5} | {time_vi:<20.4f} | {time_pi:<20.4f}")
    print(f"Iterations/Cycles{'':<6} | {iters_vi:<20} | {iters_pi:<20}")
    print(f"Success Rate (%){'':<7} | {success_rate_vi:<20.2f} | {success_rate_pi:<20.2f}")
    print(f"Avg. Path Length{'':<7} | {avg_len_vi:<20.2f} | {avg_len_pi:<20.2f}")
    print("="*50)
