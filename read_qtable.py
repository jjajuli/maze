import numpy as np

def read_q_table():
    q_table = np.load("q_table_1016-15.21.26.npy")
    print(q_table)

if __name__ == '__main__':
    read_q_table()