import Class_Train_Maze_solver
import maze_creation
import numpy as np
import random
import time

# Q-learning parameters
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.1 # Exploration rate (epsilon-greedy)
episodes = 500000  # Number of training episodes
max_steps_per_episode = 1000  # Set a limit for the number of steps per episode

def main():
    maze_file = '5x5b.csv'
    maze = maze_creation.create_maze(maze_file)
    maze_creation.print_maze(maze)
    #Accessing the element at (row,column)
    start = (0, 1)
    goal = (4, 3)
    
    env = Class_Train_Maze_solver.Class_Train_Maze_solver(maze_file, start, goal)
    q_table = initialize_q_table(maze)

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0  # Initialize the step counter for each episode
        total_reward = 0

        while not done and steps < max_steps_per_episode:
            current_state = get_state_from_position(state, maze)
            
            # Choose action using epsilon-greedy policy
            action = choose_action(current_state, q_table)
            
            # Take action and observe the result
            next_state, reward, done = env.step(action)
            next_state_id = get_state_from_position(next_state, maze)
            
            # Update the Q-table
            update_q_table(q_table, current_state, action, reward, next_state_id)
            
            total_reward += reward
            state = next_state
            steps += 1


        # while not done and steps < max_steps_per_episode:
        #     #print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        #     #env.render()  # Render the current state of the maze
            
        #     # Choose an action (you may replace this with the Q-learning decision)
        #     action = random.randint(0, 3)
            
        #     # Take the action and observe the result
        #     state, reward, done = env.step(action)
        #     total_reward += reward
        #     steps += 1  # Increment step counter

        #print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")

        if done:
            print("Goal reached!")
        else:
            print("Max steps reached. Moving to next episode.")


    print("Training complete!")

    timestr = time.strftime("%m%d-%H.%M.%S")
    np.save("q_table_"+timestr+".npy", q_table)
    env.render()

def initialize_q_table(maze):
    num_states = maze.shape[0] * maze.shape[1]
    num_actions = 4  # Up, Down, Left, Right
    return np.zeros((num_states, num_actions))

def get_state_from_position(position, maze):
    return position[0] * maze.shape[1] + position[1]

def choose_action(state, q_table):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error



if __name__ == '__main__':
    main()
