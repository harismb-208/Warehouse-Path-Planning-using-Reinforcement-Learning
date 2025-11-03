# --- visualization.py ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import config

def plot_value_and_policy(env, V, policy, title):
    """
    Generates and saves a combined plot of the value function heatmap,
    the full policy (all red arrows), and the single optimal path from S to G.
    
    NOW ALSO SHOWS THE PLOT.
    """
    value_grid = np.full((env.height, env.width), -np.inf)
    for state, value in V.items():
        value_grid[state] = value

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(value_grid, annot=False, fmt=".2f", cmap="viridis", cbar=False, ax=ax,
                linewidths=.5, linecolor='black')

    # Overlay obstacles
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle(obs[::-1], 1, 1, facecolor='gray', edgecolor='black', hatch='/'))

    # === 1. Overlay ALL policy arrows (as before) ===
    action_arrows = {'UP': (0, -0.4), 'DOWN': (0, 0.4), 'LEFT': (-0.4, 0), 'RIGHT': (0.4, 0)}
    for state, action in policy.items():
        if action in action_arrows and state != env.goal_pos:
            r, c = state
            dx, dy = action_arrows[action]
            # Plot arrow from center of the cell
            ax.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red', zorder=3)

    # === 2. Find and Highlight the Optimal Path from Start ===
    path_states = [env.start_pos]
    current_state = env.start_pos
    
    for _ in range(config.MAX_PATH_LENGTH):
        if current_state == env.goal_pos:
            break
        action = policy.get(current_state)
        if not action or action == 'GOAL':
            break
        
        move = env.actions[action]
        next_state, _ = env._calculate_next_state(current_state, move)
        path_states.append(next_state)
        current_state = next_state
        if current_state in env.obstacles:
            print("Warning: Policy led into an obstacle during path plot.")
            break
            
    # Draw the path as a thick, bright line
    if len(path_states) > 1:
        x_data = [s[1] + 0.5 for s in path_states]
        y_data = [s[0] + 0.5 for s in path_states]
        # Use a bright, thick line with markers to make it stand out
        ax.plot(x_data, y_data, 'o-', color='cyan', linewidth=4, markersize=10, 
                markeredgecolor='black', markerfacecolor='orange', zorder=4)

    # === 3. Mark Start and Goal (on top) ===
    ax.add_patch(plt.Rectangle(env.start_pos[::-1], 1, 1, facecolor='blue', alpha=0.9, zorder=5))
    ax.text(env.start_pos[1] + 0.5, env.start_pos[0] + 0.5, 'S',
            ha='center', va='center', color='white', weight='bold', fontsize=12, zorder=6)
    
    ax.add_patch(plt.Rectangle(env.goal_pos[::-1], 1, 1, facecolor='green', alpha=0.9, zorder=5))
    ax.text(env.goal_pos[1] + 0.5, env.goal_pos[0] + 0.5, 'G',
            ha='center', va='center', color='white', weight='bold', fontsize=12, zorder=6)

    # --- Final plot setup ---
    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(env.width) + 0.5)
    ax.set_yticks(np.arange(env.height) + 0.5)
    ax.set_xticklabels(np.arange(env.width))
    ax.set_yticklabels(np.arange(env.height))
    ax.tick_params(length=0) # remove ticks
    plt.gca().invert_yaxis()
    
    # Save the figure to a file
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    # *** ADDED THIS LINE ***
    # This will pause the script and show the plot
    plt.tight_layout()

    plt.show() 
    
    plt.close(fig)

def animate_robot_path(env, policy, title="Robot Path Animation"):
    """
    Creates and displays an animation of the robot following the learned policy.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def setup_grid():
        ax.clear()
        ax.set_xticks(np.arange(env.width + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(env.height + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle(obs[::-1], 1, 1, facecolor='gray'))
        ax.add_patch(plt.Rectangle(env.start_pos[::-1], 1, 1, facecolor='blue', alpha=0.5))
        ax.add_patch(plt.Rectangle(env.goal_pos[::-1], 1, 1, facecolor='green', alpha=0.5))
        ax.set_title(title)
        plt.gca().invert_yaxis()

    setup_grid()
    
    robot_patch = plt.Circle((env.start_pos[1] + 0.5, env.start_pos[0] + 0.5), 0.3, color='red', zorder=5)
    ax.add_patch(robot_patch)

    path_states = [env.start_pos]
    
    # *** FIXED TYPO HERE ***
    # It was 'current_.state' before, now it's 'current_state'
    current_state = env.start_pos
    
    # Generate the path based on the policy
    for _ in range(config.MAX_PATH_LENGTH):
        if current_state == env.goal_pos:
            break
        action = policy.get(current_state)
        if not action or action == 'GOAL':
            break
        
        move = env.actions[action]
        next_state, _ = env._calculate_next_state(current_state, move)
        path_states.append(next_state)
        current_state = next_state
        if current_state in env.obstacles:
            break
    
    path_line, = ax.plot([], [], 'o-', color='orange', linewidth=3, markersize=8, zorder=3)
    path_arrows = []

    def update(frame):
        state = path_states[frame]
        robot_patch.center = (state[1] + 0.5, state[0] + 0.5)

        x_data = [s[1] + 0.5 for s in path_states[:frame + 1]]
        y_data = [s[0] + 0.5 for s in path_states[:frame + 1]]
        path_line.set_data(x_data, y_data)
        
        for arrow in path_arrows:
            arrow.remove()
        path_arrows.clear()
        
        action_arrows_offset = {'UP': (0, -0.2), 'DOWN': (0, 0.2), 'LEFT': (-0.2, 0), 'RIGHT': (0.2, 0)}
        
        for i in range(frame):
            start_r, start_c = path_states[i]
            end_r, end_c = path_states[i+1]
            
            dr = end_r - start_r
            dc = end_c - start_c
            
            action_name = None
            if dr == -1: action_name = 'UP'
            elif dr == 1: action_name = 'DOWN'
            elif dc == -1: action_name = 'LEFT'
            elif dc == 1: action_name = 'RIGHT'

            if action_name:
                dx, dy = action_arrows_offset[action_name]
                arrow = ax.arrow(start_c + 0.5 - dx/2, start_r + 0.5 - dy/2, dx, dy,
                                 head_width=0.25, head_length=0.25, fc='cyan', ec='cyan', zorder=4)
                path_arrows.append(arrow)

        return [robot_patch, path_line] + path_arrows

    ani = animation.FuncAnimation(fig, update, frames=len(path_states),
                                  interval=config.ANIMATION_SPEED_MS, blit=True, repeat=False)
    
    plt.show()

