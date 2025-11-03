# --- main.py ---
from grid_environment import GridEnvironment
from mdp_solver import MDPSolver
from visualization import plot_value_and_policy, animate_robot_path
from evaluation import compare_algorithms

def main():
    """
    Main function to run the Warehouse Robot Path Planning simulation.
    """
    # 1. Initialize the Environment
    print("Initializing warehouse environment...")
    env = GridEnvironment()

    # 2. Initialize the MDP Solver
    solver = MDPSolver(env)

    # 3. Run Value Iteration
    vi_results = solver.value_iteration()
    policy_vi, V_vi, _, _ = vi_results

    # 4. Run Policy Iteration
    pi_results = solver.policy_iteration()
    policy_pi, V_pi, _, _ = pi_results
    
    # 5. Compare the algorithms
    # Note: Policies might be identical, comparison shows performance differences.
    compare_algorithms(vi_results, pi_results, env)
    
    # 6. Visualize the results from Value Iteration
    print("\nGenerating visualizations for Value Iteration results...")
    plot_value_and_policy(
        env, V_vi, policy_vi, 
        "Value Iteration - Optimal Value Function and Policy"
    )
    
    # --- ADDED THIS BLOCK ---
    # 6b. Visualize the results from Policy Iteration
    print("\nGenerating visualizations for Policy Iteration results...")
    plot_value_and_policy(
        env, V_pi, policy_pi, 
        "Policy Iteration - Optimal Value Function and Policy"
    )
    # --- END OF ADDED BLOCK ---
    
    # 7. Animate the robot's path (using the VI policy)
    print("Starting robot path animation...")
    animate_robot_path(env, policy_vi)

    print("\nSimulation complete.")

if __name__ == "__main__":
    main()

