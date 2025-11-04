import tkinter as tk
from tkinter import messagebox, filedialog
import json
import os
import random
from typing import List, Tuple, Set, Dict, Any

Cell = Tuple[int, int]


class InvisibleMazeGame(tk.Tk):
    def __init__(self, rows: int = 12, cols: int = 16, cell_size: int = 36):
        super().__init__()
        self.title("Invisible Maze")
        self.resizable(False, False)

        # --- Board parameters ---
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.margin = 12
        self.canvas_w = cols * cell_size + self.margin * 2
        self.canvas_h = rows * cell_size + self.margin * 2

        # --- Game state ---
        self.start: Cell = (0, 0)
        self.goal: Cell = (self.rows - 1, self.cols - 1)
        self.path: List[Cell] = []
        self.path_set: Set[Cell] = set()
        self.player: Cell = self.start
        self.falls = 0

        # Store loaded maze info when loading from file
        self.loaded_maze: Dict[str, Any] | None = None

        # --- UI ---
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        self.show_path_var = tk.BooleanVar(value=False)
        self.show_path_cb = tk.Checkbutton(
            self, text="Show path", variable=self.show_path_var, command=self.render
        )
        self.show_path_cb.grid(row=1, column=0, sticky="w", padx=(12, 0))

        self.new_btn = tk.Button(self, text="New Random Maze", command=self.new_random_maze)
        self.new_btn.grid(row=1, column=1, sticky="w")

        self.retry_btn = tk.Button(self, text="Restart (from start)", command=self.reset_player)
        self.retry_btn.grid(row=1, column=2, sticky="w")

        self.load_btn = tk.Button(self, text="Load Maze", command=self.load_maze_from_file)
        self.load_btn.grid(row=1, column=3, sticky="w")

        self.falls_var = tk.StringVar(value="Falls: 0")
        self.falls_lbl = tk.Label(self, textvariable=self.falls_var, font=("Helvetica", 12, "bold"))
        self.falls_lbl.grid(row=1, column=4, sticky="e", padx=(0, 12))

        # Grid setup
        self.rect_by_cell: Dict[Cell, int] = {}
        self.build_grid_rects()

        # Keyboard controls
        self.bind_all("<Up>", lambda e: self.try_move((-1, 0)))
        self.bind_all("<Down>", lambda e: self.try_move((1, 0)))
        self.bind_all("<Left>", lambda e: self.try_move((0, -1)))
        self.bind_all("<Right>", lambda e: self.try_move((0, 1)))
        self.bind_all("w", lambda e: self.try_move((-1, 0)))
        self.bind_all("s", lambda e: self.try_move((1, 0)))
        self.bind_all("a", lambda e: self.try_move((0, -1)))
        self.bind_all("d", lambda e: self.try_move((0, 1)))

        # Start with a random maze
        self.new_random_maze()

    # ---------------- Grid Construction ---------------- #
    def build_grid_rects(self):
        """Build or rebuild the canvas rectangles according to the grid size."""
        self.canvas_w = self.cols * self.cell_size + self.margin * 2
        self.canvas_h = self.rows * self.cell_size + self.margin * 2
        self.canvas.config(width=self.canvas_w, height=self.canvas_h)

        self.canvas.delete("all")
        self.rect_by_cell.clear()

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.margin + c * self.cell_size
                y0 = self.margin + r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#000", outline="#222")
                self.rect_by_cell[(r, c)] = rect

        self.update_idletasks()

    def apply_board_size(self, rows: int, cols: int, cell_size: int | None = None):
        """Change grid dimensions and rebuild the board."""
        self.rows = int(rows)
        self.cols = int(cols)
        if cell_size is not None:
            self.cell_size = int(cell_size)
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.player = self.start
        self.build_grid_rects()

    # ---------------- Random Maze ---------------- #
    def new_random_maze(self):
        """Generate a random maze and clear any previously loaded maze."""
        self.loaded_maze = None
        self.falls = 0
        self.apply_board_size(self.rows, self.cols, self.cell_size)
        self.generate_path_random()
        self.update_falls_label()
        self.render()

    def reset_player(self):
        self.player = self.start
        self.render()

    def generate_path_random(self):
        """Generate a valid path using random DFS."""
        self.path = self._dfs_path(self.start, self.goal)
        tries = 0
        while not self.path and tries < 10:
            tries += 1
            self.path = self._dfs_path(self.start, self.goal)
        if not self.path:
            # Fallback: simple L-shape
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
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set([start])
        parent: Dict[Cell, Cell] = {}
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == goal:
                break
            r, c = cur
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols and nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    stack.append(nxt)
        if goal not in parent and goal != start:
            return []
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # ---------------- Load Maze from File ---------------- #
    def load_maze_from_file(self):
        """Load a maze (JSON) created with the path editor."""
        filepath = filedialog.askopenfilename(
            title="Select a maze file",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=os.path.abspath("paths") if os.path.isdir("paths") else os.getcwd(),
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_loaded_maze(data)
            messagebox.showinfo("Maze loaded", f"Maze successfully loaded from:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    def apply_loaded_maze(self, data: Dict[str, Any]):
        """Apply dimensions, start/goal, and path from the JSON file."""
        rows = int(data.get("rows", self.rows))
        cols = int(data.get("cols", self.cols))
        cell_size = int(data.get("cell_size", self.cell_size))
        start_list = data.get("start", [0, 0])
        goal_list = data.get("goal", [rows - 1, cols - 1])
        path_list = data.get("path", [])

        self.apply_board_size(rows, cols, cell_size)

        sr, sc = int(start_list[0]), int(start_list[1])
        gr, gc = int(goal_list[0]), int(goal_list[1])
        sr = max(0, min(self.rows - 1, sr))
        sc = max(0, min(self.cols - 1, sc))
        gr = max(0, min(self.rows - 1, gr))
        gc = max(0, min(self.cols - 1, gc))
        self.start = (sr, sc)
        self.goal = (gr, gc)
        self.player = self.start

        path: List[Cell] = []
        for rc in path_list:
            if isinstance(rc, list) and len(rc) == 2:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    path.append((r, c))
        if self.start not in path:
            path.insert(0, self.start)
        if self.goal not in path:
            path.append(self.goal)
        self.path = path
        self.path_set = set(path)

        self.loaded_maze = {
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
            "start": self.start,
            "goal": self.goal,
            "path": [list(c) for c in self.path],
        }
        self.falls = 0
        self.update_falls_label()
        self.render()

    # ---------------- Movement and Game Logic ---------------- #
    def try_move(self, delta: Tuple[int, int]):
        nr = self.player[0] + delta[0]
        nc = self.player[1] + delta[1]
        nxt = (nr, nc)
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            self.register_fall()
            return
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
        if self.loaded_maze is not None:
            self.apply_loaded_maze(self.loaded_maze)
        else:
            self.new_random_maze()

    def update_falls_label(self):
        self.falls_var.set(f"Falls: {self.falls}")

    # ---------------- Rendering ---------------- #
    def render(self):
        show_path = self.show_path_var.get()
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                rect = self.rect_by_cell.get(cell)
                if rect is None:
                    continue
                fill = "#000"
                outline = "#222"
                if show_path and cell in self.path_set:
                    fill = "#2b2b2b"
                if cell == self.start:
                    fill = "#2ecc71"
                if cell == self.goal:
                    fill = "#f1c40f"
                if cell == self.player:
                    fill = "#3498db"
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
    app = InvisibleMazeGame(rows=12, cols=16, cell_size=36)
    app.mainloop()


if __name__ == "__main__":
    main()
