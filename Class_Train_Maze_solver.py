
import maze_creation
import numpy as np

# Define a simple maze environment using a grid
class Class_Train_Maze_solver:
    def __init__(self,file_path,start,goal):
        self.maze = maze_creation.create_maze(file_path)
        self.max_colums = (self.maze.shape[0] )- 1
        self.max_rows = (self.maze.shape[1]) - 1
        #print (maze_creation.get_path_list(self.maze))

        # Define the start and goal positions
        self.file_path = file_path
        self.start = start
        self.goal = goal
        # Initialize agent's position
        self.agent_pos = list(self.start)
        # Step counter
        self.steps = 0  # Initialize step counter

        
    
    def reset(self):
        """Reset the environment and place the agent at the start."""
        self.agent_pos = list(self.start)
        self.steps = 0  # Reset the step counter
        return self.agent_pos
    
    def step(self, action):
        """
        Take an action and move the agent.
        Actions are: 0 = up, 1 = down, 2 = left, 3 = right.
        """
        # Get the proposed new position
        new_pos = list(self.agent_pos)
        if action == 0:   # Up
            new_pos[0] -= 1
        elif action == 1: # Down
            new_pos[0] += 1
        elif action == 2: # Left
            new_pos[1] -= 1
        elif action == 3: # Right
            new_pos[1] += 1
        
        # Check if the new position is out of bounds or an obstacle
        if (0 <= new_pos[0] < self.max_colums) and (0 <= new_pos[1] < self.max_rows) and (self.maze[new_pos[0], new_pos[1]] == 0):
            self.agent_pos = new_pos
            self.steps += 1  # Increment the step counter
        
        # Check if the agent has reached the goal
        done = (self.agent_pos == list(self.goal))
        
        # Return the new state, a reward, and whether the game is done
        """ **  1  **  Reward is 1 if the agent reaches the goal, -1 otherwise."""        
        # reward = 1 if done else -1  # Small negative reward for each step, 1 for reaching the goal
        # print(f"new pos {new_pos} - Reward: {reward}, done: {done}")

        """ **  2  **  Reward is 0.1 if the agent move and 1 if reaches the goal, -1 otherwise."""        
        # check if new pos is in the path list
        
        path_list = maze_creation.get_path_list(self.maze)
        temp_new_pos = tuple(new_pos)
        if temp_new_pos in path_list:
            reward = 0.2
        else:
            reward = -1
        if done:
            reward = 1
            print(f"Steps: {self.steps}")



        #print(f"new pos {new_pos} - Reward: {reward}, done: {done}")

         

        return self.agent_pos, reward, done
    
    def render(self):
        """Print the maze and the agent's position."""
        maze_copy = self.maze.copy()
        maze_copy[self.agent_pos[0], self.agent_pos[1]] = 2  # Mark the agent's position
        for row in maze_copy:
            print("".join(["A" if cell == 2 else "█" if cell == 1 else " " for cell in row]))
        



