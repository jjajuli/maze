# This code read a csv file and create a maze from it
# The maze is a 2D array of 0s and 1s
# 1s are the walls and 0s are the paths


import csv
import numpy as np

def create_maze(file_path):
    maze = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            maze.append([int(cell) for cell in row])
    maze_array = np.array(maze)
    return maze_array

def print_maze(maze):
     for row in maze:
        # Convert each row into a string of ASCII characters
        ascii_row = ''.join(['*' if cell == 1 else '█' for cell in row])
        print(ascii_row)
    


