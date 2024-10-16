import numpy as np
import Class_Train_Maze_solver

def solve_maze(env, q_table):
    state = env.reset()
    done = False
    steps = 0
    max_steps = 1000

    while not done and steps < max_steps:
        print(f"Steps {steps + 1}")
        env.render()
        # Get the state ID from the agent's position
        current_state = get_state_from_position(state, env.maze)
        # Choose the best action from the Q-table
        action = np.argmax(q_table[current_state])
        # Take the action
        state, reward, done = env.step(action)
        steps += 1

    print(f"Goal reached in {steps} steps!")

def get_state_from_position(position, maze):
    return position[0] * maze.shape[1] + position[1]

def main():
    # Create the maze environment
    maze_file = "5x5b.csv"
    start = (0, 1)
    goal = (4, 3)
    env = Class_Train_Maze_solver.Class_Train_Maze_solver(maze_file, start, goal)

    # Load the Q-table
    q_table = np.load("q_table_1016-16.51.28.npy")

    # Solve the maze
    solve_maze(env, q_table)

if __name__ == '__main__':
    main()