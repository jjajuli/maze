import argparse
import json
import os
import sys
import time
from typing import Optional

# --- Ajusta este import según dónde esté definida tu clase MazeGame ---
# Si tu MazeGame vive en otro archivo (p.ej. maze_gui.py), cámbialo a:
# from maze_gui import MazeGame
from maze import MazeGame  # <-- Si pegas MazeGame arriba de este archivo, funcionará.
# ----------------------------------------------------------------------

import numpy as np

# Cargamos algoritmos comunes de SB3 y un cargador "agnóstico"
from stable_baselines3 import PPO, DQN, A2C, TD3, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm

ALGOS_TO_TRY = [PPO, DQN, A2C, TD3, SAC, DDPG]


def load_model_any(model_path: str, env=None) -> BaseAlgorithm:
    """
    Intenta cargar el .zip con varios algoritmos de SB3 sin que el usuario
    tenga que especificar cuál fue.
    """
    last_err = None
    for Algo in ALGOS_TO_TRY:
        try:
            # Nota: pasar env aquí permite usar predict sin crear VecEnv aparte
            return Algo.load(model_path, env=env, print_system_info=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude cargar el modelo desde {model_path}. "
                       f"¿Es un zip entrenado con SB3? Último error: {last_err}")


def normalized_obs_from_game(game: MazeGame) -> np.ndarray:
    """Observación igual que en el entorno de entrenamiento: (fila, col) normalizada a [0,1]."""
    r, c = game.player
    rows = max(1, game.rows - 1)
    cols = max(1, game.cols - 1)
    return np.array([r / rows, c / cols], dtype=np.float32)


def apply_action_to_game(game: MazeGame, action: int) -> dict:
    """
    Aplica la acción al MazeGame sin disparar diálogos.
    Devuelve info con flags: fell (bool), reached_goal (bool).
    Convención de acciones: 0=up, 1=down, 2=left, 3=right.
    """
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dr, dc = deltas[int(action)]
    nr = game.player[0] + dr
    nc = game.player[1] + dc
    nxt = (nr, nc)

    # Chequeo de límites y pertenencia al camino
    if not (0 <= nr < game.rows and 0 <= nc < game.cols) or nxt not in game.path_set:
        # “Caída”: usa la propia lógica visual/estado del juego
        game.register_fall()
        return {"fell": True, "reached_goal": False}

    # Movimiento válido
    game.player = nxt
    game.render()

    if game.player == game.goal:
        # NO llamamos a on_win() (evitamos messagebox + reinicio automático)
        return {"fell": False, "reached_goal": True}

    return {"fell": False, "reached_goal": False}


class TkPolicyRunner:
    """
    Ejecuta una política SB3 sobre el MazeGame, paso a paso, con animación en Tk.
    """

    def __init__(self, game: MazeGame, model: BaseAlgorithm, delay_ms: int = 120, max_steps: Optional[int] = None):
        self.game = game
        self.model = model
        self.delay_ms = max(1, int(delay_ms))
        self.max_steps = max_steps if max_steps is not None else (game.rows * game.cols * 4)
        self._running = False
        self._steps = 0

        # bindings opcionales: espacio=toggle play/pause | r=reset | q=quit
        self.game.bind_all("<space>", lambda e: self.toggle())
        self.game.bind_all("r", lambda e: self.reset())
        self.game.bind_all("q", lambda e: self.quit())

    def start(self):
        if self._running:
            return
        self._running = True
        self._steps = 0
        self._tick()

    def toggle(self):
        self._running = not self._running
        if self._running:
            self._tick()

    def reset(self):
        # Reseteo “manual”: volver a start y redibujar
        self.game.player = self.game.start
        self.game.render()
        self._steps = 0

    def quit(self):
        self._running = False
        self.game.destroy()

    def _tick(self):
        if not self._running:
            return

        # Paradas por seguridad
        if self._steps >= self.max_steps:
            print("Máximo de pasos alcanzado. Deteniendo reproducción.")
            self._running = False
            return

        obs = normalized_obs_from_game(self.game)
        action, _ = self.model.predict(obs, deterministic=True)
        info = apply_action_to_game(self.game, int(action))
        self._steps += 1

        if info["reached_goal"]:
            print(f"¡Meta alcanzada en {self._steps} pasos! Caídas totales mostradas en la UI.")
            # Aquí podrías hacer un pequeño flash/pausa y detener.
            self._running = False
            return

        # Programa el siguiente paso
        self.game.after(self.delay_ms, self._tick)


def load_maze_json_into_game(game: MazeGame, json_path: Optional[str]):
    """Carga un laberinto JSON con el mismo formato que usa tu editor y lo aplica al MazeGame."""
    if not json_path:
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    game.apply_loaded_maze(data)


def main():
    parser = argparse.ArgumentParser(description="Viewer Tk para reproducir un modelo SB3 dentro de MazeGame")
    parser.add_argument("--model", required=True, help="Ruta al .zip entrenado (SB3)")
    parser.add_argument("--json", default=None, help="(Opcional) Ruta al laberinto JSON a cargar en MazeGame")
    parser.add_argument("--rows", type=int, default=12, help="Filas por defecto si no hay JSON")
    parser.add_argument("--cols", type=int, default=16, help="Columnas por defecto si no hay JSON")
    parser.add_argument("--cell", type=int, default=36, help="Tamaño de celda (px)")
    parser.add_argument("--delay", type=int, default=120, help="Delay entre pasos (ms) para animación")
    parser.add_argument("--max_steps", type=int, default=None, help="Corte de seguridad de pasos")
    args = parser.parse_args()

    # Crea la ventana con tu MazeGame
    app = MazeGame(rows=args.rows, cols=args.cols, cell_size=args.cell)

    # Si hay JSON, ajusta el tablero y el camino cargado
    if args.json:
        load_maze_json_into_game(app, args.json)

    # Carga del modelo (auto-detectando el algoritmo)
    # No necesitamos un "env" real, ya que solo usamos predict() con la observación actual.
    model = load_model_any(args.model, env=None)

    # Arranca el runner con animación
    runner = TkPolicyRunner(app, model, delay_ms=args.delay, max_steps=args.max_steps)

    # Botón extra en la UI para reproducir/pausar
    import tkinter as tk
    ctrl_frame = tk.Frame(app)
    ctrl_frame.grid(row=2, column=0, columnspan=5, pady=(0, 10))

    play_btn = tk.Button(ctrl_frame, text="▶ Reproducir", command=runner.start)
    pause_btn = tk.Button(ctrl_frame, text="⏸ Pausa", command=runner.toggle)
    reset_btn = tk.Button(ctrl_frame, text="↺ Reset", command=runner.reset)
    quit_btn = tk.Button(ctrl_frame, text="✕ Salir", command=runner.quit)
    play_btn.grid(row=0, column=0, padx=4)
    pause_btn.grid(row=0, column=1, padx=4)
    reset_btn.grid(row=0, column=2, padx=4)
    quit_btn.grid(row=0, column=3, padx=4)

    # Arranca el mainloop de Tk
    app.mainloop()


if __name__ == "__main__":
    main()
