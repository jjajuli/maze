import tkinter as tk
from tkinter import messagebox, filedialog
import os
import json
from datetime import datetime
from typing import List, Tuple, Set, Dict, Any

Cell = Tuple[int, int]

class PathEditorGame(tk.Tk):
    """
    Path Editor version:
    - The player defines the path by moving around.
    - A gray trail marks the visited cells.
    - Start (green) and Goal (yellow) are always visible.
    - When reaching the goal, the user is asked if they want to save the path.
      If yes, the path is saved as ./paths/path_YYYYmmdd_HHMMSS.json

    JSON file format:
    {
      "rows": int,
      "cols": int,
      "cell_size": int,
      "start": [r, c],
      "goal": [r, c],
      "path": [[r0, c0], [r1, c1], ...],  # in chronological order
      "created_at": "ISO-8601 UTC timestamp"
    }
    """

    def __init__(self, rows: int = 12, cols: int = 16, cell_size: int = 36):
        super().__init__()
        self.title("Path Editor - Invisible Maze")
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
        self.player: Cell = self.start
        self.trail: List[Cell] = [self.start]  # chronological movement list
        self.trail_set: Set[Cell] = {self.start}

        # --- UI ---
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        self.new_btn = tk.Button(self, text="New (clear)", command=self.reset_board)
        self.new_btn.grid(row=1, column=0, sticky="w", padx=(12, 0))

        self.restart_btn = tk.Button(self, text="Restart at start", command=self.restart_player)
        self.restart_btn.grid(row=1, column=1, sticky="w")

        self.save_btn = tk.Button(self, text="Save path", command=self.save_path)
        self.save_btn.grid(row=1, column=2, sticky="w")

        self.load_btn = tk.Button(self, text="Load and draw path", command=self.load_and_draw_path)
        self.load_btn.grid(row=1, column=3, sticky="w")

        self.info_lbl = tk.Label(self, text="Move with arrows/WASD. Reach the goal to save.")
        self.info_lbl.grid(row=1, column=4, sticky="e", padx=(0, 12))

        # Create grid rectangles
        self.rect_by_cell: Dict[Cell, int] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.margin + c * self.cell_size
                y0 = self.margin + r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#000", outline="#222")
                self.rect_by_cell[(r, c)] = rect

        # Keyboard controls (arrows + WASD)
        self.bind_all("<Up>",    lambda e: self.try_move((-1, 0)))
        self.bind_all("<Down>",  lambda e: self.try_move((1, 0)))
        self.bind_all("<Left>",  lambda e: self.try_move((0, -1)))
        self.bind_all("<Right>", lambda e: self.try_move((0, 1)))
        self.bind_all("w", lambda e: self.try_move((-1, 0)))
        self.bind_all("s", lambda e: self.try_move((1, 0)))
        self.bind_all("a", lambda e: self.try_move((0, -1)))
        self.bind_all("d", lambda e: self.try_move((0, 1)))

        self.render()

    # ----------------- Main logic ----------------- #
    def reset_board(self):
        self.player = self.start
        self.trail = [self.start]
        self.trail_set = {self.start}
        self.render()

    def restart_player(self):
        self.player = self.start
        if self.start not in self.trail_set:
            self.trail.append(self.start)
            self.trail_set.add(self.start)
        self.render()

    def try_move(self, delta: Tuple[int, int]):
        nr = self.player[0] + delta[0]
        nc = self.player[1] + delta[1]
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # Ignore moves that go outside the board
            return
        nxt = (nr, nc)
        self.player = nxt
        # Add to trail (track all visited cells)
        if nxt not in self.trail_set:
            self.trail.append(nxt)
            self.trail_set.add(nxt)
        else:
            # Add duplicates to maintain full movement sequence
            self.trail.append(nxt)
        self.render()
        if self.player == self.goal:
            self.on_reach_goal()

    def on_reach_goal(self):
        ans = messagebox.askyesno("Goal reached", "Do you want to save this path with a timestamp?")
        if ans:
            self.save_path()

    # ----------------- Save / Load ----------------- #
    def build_path_payload(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
            "start": [self.start[0], self.start[1]],
            "goal": [self.goal[0], self.goal[1]],
            "path": [[r, c] for (r, c) in self.trail],
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def save_path(self):
        os.makedirs("paths", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"path_{ts}.json"
        fullpath = os.path.join("paths", filename)
        payload = self.build_path_payload()
        try:
            with open(fullpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Path saved to:\n{fullpath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save the file:\n{e}")

    def load_and_draw_path(self):
        filepath = filedialog.askopenfilename(
            title="Select a path file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.abspath("paths") if os.path.isdir("paths") else os.getcwd()
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_loaded_path(data)
            messagebox.showinfo("Loaded", f"Path loaded from:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load the file:\n{e}")

    def apply_loaded_path(self, data: Dict[str, Any]):
        loaded_path = data.get("path", [])
        if not loaded_path:
            return
        self.reset_board()
        for rc in loaded_path:
            if not isinstance(rc, list) or len(rc) != 2:
                continue
            r, c = rc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.player = (r, c)
                if self.player not in self.trail_set:
                    self.trail.append(self.player)
                    self.trail_set.add(self.player)
                else:
                    self.trail.append(self.player)
        self.render()

    # ----------------- Rendering ----------------- #
    def render(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                rect = self.rect_by_cell[cell]
                fill = "#000"
                outline = "#222"
                # Trail
                if cell in self.trail_set:
                    fill = "#2b2b2b"  # gray trail
                # Start and Goal
                if cell == self.start:
                    fill = "#2ecc71"  # green
                if cell == self.goal:
                    fill = "#f1c40f"  # yellow
                # Player on top
                if cell == self.player:
                    fill = "#3498db"  # blue
                    outline = "#5faee3"
                self.canvas.itemconfig(rect, fill=fill, outline=outline)
        self.canvas.update_idletasks()


def main():
    app = PathEditorGame(rows=12, cols=16, cell_size=36)
    app.mainloop()


if __name__ == "__main__":
    main()
