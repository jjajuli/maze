import csv
import numpy as np
import maze_creation

    
import numpy as np

# Define a simple maze environment using a grid
class SimpleMaze:
    def __init__(self,file_path,start,goal):
        maze_creation.create_maze(file_path)

        # Define the start and goal positions
        self.start = start
        self.goal = goal
        # Initialize agent's position
        self.agent_pos = list(self.start)
    
    def reset(self):
        """Reset the environment and place the agent at the start."""
        self.agent_pos = list(self.start)
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
        if (0 <= new_pos[0] < 5) and (0 <= new_pos[1] < 5) and (self.maze[new_pos[0], new_pos[1]] == 0):
            self.agent_pos = new_pos
        
        # Check if the agent has reached the goal
        done = (self.agent_pos == list(self.goal))
        
        # Return the new state, a reward, and whether the game is done
        reward = 1 if done else -0.1  # Small negative reward for each step, 1 for reaching the goal
        return self.agent_pos, reward, done
    
    def render(self):
        """Print the maze and the agent's position."""
        maze_copy = self.maze.copy()
        maze_copy[self.agent_pos[0], self.agent_pos[1]] = 2  # Mark the agent's position
        for row in maze_copy:
            print(" ".join(["A" if cell == 2 else "X" if cell == 1 else "." for cell in row]))
        print("\n")

# Initialize and test the environment
env = SimpleMaze()
state = env.reset()
env.render()

# Example move (move down)
state, reward, done = env.step(1)
env.render()



def main():
    file_path = '5x5b.csv'
    maze = maze_creation.create_maze(file_path)
    maze_creation.print_maze(maze)
    start = (1, 0)
    goal = (3, 4)
    SimpleMaze(file_path, start, goal)


if __name__ == '__main__':
    main()