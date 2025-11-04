import tkinter as tk
from tkinter import messagebox
import random
from typing import List, Tuple, Set

Cell = Tuple[int, int]

class InvisibleMazeGame(tk.Tk):
    def __init__(self, rows: int = 12, cols: int = 16, cell_size: int = 36):
        super().__init__()
        self.title("Invisible Maze")
        self.resizable(False, False)

        # --- Grid configuration ---
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.margin = 12
        self.canvas_w = cols * cell_size + self.margin * 2
        self.canvas_h = rows * cell_size + self.margin * 2

        # --- Game state ---
        self.start: Cell = (0, 0)
        self.goal: Cell = (self.rows - 1, self.cols - 1)
        self.path: List[Cell] = []          # path from start to goal
        self.path_set: Set[Cell] = set()    # for O(1) lookup
        self.player: Cell = self.start
        self.falls = 0

        # --- UI setup ---
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.show_path_var = tk.BooleanVar(value=False)
        self.show_path_cb = tk.Checkbutton(
            self, text="Show path", variable=self.show_path_var, command=self.render
        )
        self.show_path_cb.grid(row=1, column=0, sticky="w", padx=(12, 0))

        self.new_maze_btn = tk.Button(self, text="New Maze", command=self.new_maze)
        self.new_maze_btn.grid(row=1, column=1, sticky="w")

        self.retry_btn = tk.Button(self, text="Restart (from start)", command=self.reset_player)
        self.retry_btn.grid(row=1, column=2, sticky="w")

        self.falls_var = tk.StringVar(value="Falls: 0")
        self.falls_lbl = tk.Label(self, textvariable=self.falls_var, font=("Helvetica", 12, "bold"))
        self.falls_lbl.grid(row=1, column=3, sticky="e", padx=(0, 12))

        # --- Draw grid rectangles ---
        self.rect_by_cell = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.margin + c * self.cell_size
                y0 = self.margin + r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#000", outline="#222")
                self.rect_by_cell[(r, c)] = rect

        # --- Keyboard controls ---
        self.bind_all("<Up>", lambda e: self.try_move((-1, 0)))
        self.bind_all("<Down>", lambda e: self.try_move((1, 0)))
        self.bind_all("<Left>", lambda e: self.try_move((0, -1)))
        self.bind_all("<Right>", lambda e: self.try_move((0, 1)))
        # Optional WASD keys
        self.bind_all("w", lambda e: self.try_move((-1, 0)))
        self.bind_all("s", lambda e: self.try_move((1, 0)))
        self.bind_all("a", lambda e: self.try_move((0, -1)))
        self.bind_all("d", lambda e: self.try_move((0, 1)))

        # Generate and draw the first maze
        self.new_maze()

    # ---------------- Maze logic ---------------- #
    def new_maze(self):
        self.falls = 0
        self.player = self.start
        self.generate_path()
        self.update_falls_label()
        self.render()

    def reset_player(self):
        self.player = self.start
        self.render()

    def generate_path(self):
        """
        Generate a single path (no walls): valid cells suspended over the void.
        Uses random DFS to ensure the goal is reachable.
        """
        self.path = self._dfs_path(self.start, self.goal)
        tries = 0
        while not self.path and tries < 10:
            tries += 1
            self.path = self._dfs_path(self.start, self.goal)
        if not self.path:
            # fallback: simple L-shaped path
            self.path = []
            r, c = self.start
            while c < self.goal[1]:
                self.path.append((r, c))
                c += 1
            while r < self.goal[0]:
                self.path.append((r, c))
                r += 1
            self.path.append(self.goal)
        self.path_set = set(self.path)

    def _dfs_path(self, start: Cell, goal: Cell) -> List[Cell]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        parent = {}
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            if current == goal:
                break
            r, c = current
            random.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols and nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = current
                    stack.append(nxt)
        if goal not in parent and goal != start:
            return []
        # reconstruct path
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # ---------------- Movement ---------------- #
    def try_move(self, delta: Tuple[int, int]):
        nr = self.player[0] + delta[0]
        nc = self.player[1] + delta[1]
        nxt = (nr, nc)
        # Out of bounds = fall
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            self.register_fall()
            return
        # Only move if on valid path
        if nxt in self.path_set:
            self.player = nxt
            self.render()
            if self.player == self.goal:
                self.on_win()
        else:
            self.register_fall()

    def register_fall(self):
        self.falls += 1
        self.update_falls_label()
        self.flash_cell(self.player, color="#a00")
        self.player = self.start
        self.render()

    def on_win(self):
        messagebox.showinfo("You Win!", f"You reached the goal. Total falls: {self.falls}")
        self.new_maze()

    def update_falls_label(self):
        self.falls_var.set(f"Falls: {self.falls}")

    # ---------------- Rendering ---------------- #
    def render(self):
        show_path = self.show_path_var.get()
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                rect = self.rect_by_cell[cell]
                fill = "#000"
                outline = "#222"
                if show_path and cell in self.path_set:
                    fill = "#2b2b2b"  # visible path (gray)
                if cell == self.start:
                    fill = "#2ecc71"   # green start
                if cell == self.goal:
                    fill = "#f1c40f"   # yellow goal
                if cell == self.player:
                    fill = "#3498db"   # blue player
                    outline = "#5faee3"
                self.canvas.itemconfig(rect, fill=fill, outline=outline)
        self.canvas.update_idletasks()

    def flash_cell(self, cell: Cell, color="#a00", flashes: int = 1, delay_ms: int = 60):
        rect = self.rect_by_cell.get(cell)
        if not rect:
            return
        original = self.canvas.itemcget(rect, "fill")
        for _ in range(flashes):
            self.canvas.itemconfig(rect, fill=color)
            self.canvas.update_idletasks()
            self.after(delay_ms)
            self.canvas.itemconfig(rect, fill=original)
            self.canvas.update_idletasks()
            self.after(delay_ms)


def main():
    # You can adjust the grid size here
    app = InvisibleMazeGame(rows=4, cols=6, cell_size=36)
    app.mainloop()

if __name__ == "__main__":
    main()
