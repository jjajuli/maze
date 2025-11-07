import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional, Callable
import argparse
import os
import time
import re

Cell = Tuple[int, int]

"""
Cómo usarlo

# Entrenar con un laberinto aleatorio (sin Tk, rápido):
python maze_rlv2.py --train --algo PPO --steps 200000

# Entrenar usando tu archivo JSON:
python maze_rlv2.py --train --algo DQN --steps 300000 --json paths/path_20251104_145523.json

# Elegir una función de recompensa (por ID):
python maze_rlv2.py --train --algo PPO --reward_id 2 --steps 200000 --json paths/path_20251104_145523.json

# Evaluar el modelo (con render ASCII en consola):
python maze_rlv2.py --eval --model model_PPO_01_path_20251104_145523.zip --json paths/path_20251104_145523.json --render

# Reproducir modelo sobre un laberinto JSON específico
python viewer_tk.py --model model_DQN_01_path7x7_20251106_154816.zip --json .\paths\path7x7_20251106_154816.json

# O sin JSON (usa filas/cols por defecto)
python viewer_tk.py --model maze_model.zip --rows 12 --cols 16 --cell 36

"""

# -----------------------------
#  Utilidades
# -----------------------------
def safe_stem(path: Optional[str]) -> str:
    if not path:
        return "RANDOM"
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    # sanea para nombres de archivo
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)
    return stem or "RANDOM"

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -----------------------------
#  MazeCore: lógica sin GUI (física solamente)
# -----------------------------
class MazeCore:
    """
    Lógica del laberinto: estado, movimiento y finalización.
    Compatible con el formato JSON de tu editor:
    {
      "rows": int, "cols": int, "cell_size": int,
      "start": [r,c], "goal": [r,c],
      "path": [[r,c], ...]
    }
    """
    def __init__(self, rows=7, cols=7, start=(0,0), goal=None, path: Optional[List[Cell]]=None):
        self.rows = rows
        self.cols = cols
        self.start: Cell = tuple(start)
        self.goal: Cell  = (rows-1, cols-1) if goal is None else tuple(goal)
        self.player: Cell = self.start
        self.path: List[Cell] = path or self._default_L_path()
        self.path_set = set(self.path)

        # Seguridad: incluir start/goal
        if self.start not in self.path_set:
            self.path.insert(0, self.start); self.path_set.add(self.start)
        if self.goal not in self.path_set:
            self.path.append(self.goal); self.path_set.add(self.goal)

    @staticmethod
    def from_json_file(filepath: str) -> "MazeCore":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = int(data.get("rows"))
        cols = int(data.get("cols"))
        start = tuple(map(int, data.get("start", [0,0])))
        goal  = tuple(map(int, data.get("goal", [rows-1, cols-1])))
        raw_path = data.get("path", [])
        path: List[Cell] = []
        for rc in raw_path:
            if isinstance(rc, list) and len(rc) == 2:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < rows and 0 <= c < cols:
                    path.append((r,c))
        return MazeCore(rows=rows, cols=cols, start=start, goal=goal, path=path)

    def reset(self) -> Cell:
        self.player = self.start
        return self.player

    def step_physics(self, action: int) -> Dict[str, Any]:
        """
        Solo actualiza estado y devuelve info de la transición (sin recompensas).
        Acciones: 0=up,1=down,2=left,3=right
        """
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]
        dr, dc = deltas[int(action)]
        prev = self.player # posición Actual antes del paso
        nr = prev[0] + dr  # nueva fila (new row) = fila anterior + cambio de fila
        nc = prev[1] + dc  # nueva columna (new col) = columna anterior + cambio de columna
        nxt = (nr, nc)

        fell = False
        moved = False
        reached_goal = False

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            fell = True
            # reinicio por caída
            self.player = self.start
        elif nxt not in self.path_set:
            fell = True
            self.player = self.start
        else:
            # movimiento válido
            self.player = nxt
            moved = True
            if self.player == self.goal:
                reached_goal = True

        info = {
            "prev": prev,
            "curr": self.player,
            "fell": fell,
            "moved": moved,
            "reached_goal": reached_goal,
            "dist_prev": manhattan(prev, self.goal),
            "dist_curr": manhattan(self.player, self.goal),
        }
        return info

    def _default_L_path(self) -> List[Cell]:
        path = []
        r, c = self.start
        while c < self.goal[1]:
            path.append((r,c)); c += 1
        while r < self.goal[0]:
            path.append((r,c)); r += 1
        path.append(self.goal)
        return path

    def render_ascii(self) -> str:
        grid = [[" ." for _ in range(self.cols)] for _ in range(self.rows)]
        for (r,c) in self.path_set:
            grid[r][c] = "  "
        sr, sc = self.start; grid[sr][sc] = " S"
        gr, gc = self.goal;  grid[gr][gc] = " G"
        pr, pc = self.player; grid[pr][pc] = " P"
        return "\n".join("".join(row) for row in grid)


# -----------------------------
#  Recompensas (seleccionables)
# -----------------------------
# Firma: reward_fn(info: dict, step_idx: int) -> float
# info = {prev, curr, fell, moved, reached_goal, dist_prev, dist_curr}

def reward_v1_simple(info: Dict[str, Any], step_idx: int) -> float:
    """Base: +1 al llegar; -0.1 por caída; -0.01 por paso."""
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return -0.1
    return -0.01

def reward_v2_progress(info: Dict[str, Any], step_idx: int) -> float:
    """Shaping por progreso Manhattan hacia la meta."""
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return -0.1
    delta = (info["dist_prev"] - info["dist_curr"])  # >0 si se acerca
    return 0.02 * delta - 0.01  # premio por acercarse + pequeño coste por paso

def reward_v3_sparse(info: Dict[str, Any], step_idx: int) -> float:
    """Muy escasa: solo +1 al llegar; leve castigo por tiempo; caída más cara."""
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return -0.2
    return -0.02

def reward_v4_anti_loop(info: Dict[str, Any], step_idx: int) -> float:
    """Progreso + anti-bucle suave (penaliza no moverse)."""
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return -0.1
    moved = info["moved"]
    delta = (info["dist_prev"] - info["dist_curr"])
    base = 0.02 * delta - 0.01
    if not moved:  # no debería ocurrir salvo errores, pero por si acaso
        base -= 0.02
    return base

def reward_v5_strong(info: Dict[str, Any], step_idx: int) -> float:
    """
    +1 cuando da un paso válido,
    -100 cuando se cae (fuera del camino o del tablero),
    +1000 cuando llega a la meta.
    """
    if info["reached_goal"]:
        return 1000.0
    if info["fell"]:
        return -100.0
    # paso válido sin llegar a la meta
    return 1.0


# Mapa de IDs a funciones
REWARD_FNS: Dict[int, Callable[[Dict[str, Any], int], float]] = {
    1: reward_v1_simple,
    2: reward_v2_progress,
    3: reward_v3_sparse,
    4: reward_v4_anti_loop,
    5: reward_v5_strong,
}

# Preferencias por algoritmo (puedes cambiarlas libremente)
DEFAULT_REWARD_BY_ALGO: Dict[str, int] = {
    "PPO": 2,  # progreso suele ayudar a PPO
    "DQN": 1,  # simple va bien con DQN de arranque
    "A2C": 2,
    "SAC": 2,
    "TD3": 2,
    "DDPG": 2,
    "TQC": 2,
}

# -----------------------------
#  Gymnasium Env
# -----------------------------
class MazeEnv(gym.Env):
    """
    Observación: (fila, col) normalizada [0,1], shape=(2,)
    Acciones: Discrete(4) -> up,down,left,right
    Termina al llegar a la meta o por límite de pasos.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        core: MazeCore,
        reward_fn: Callable[[Dict[str, Any], int], float],
        max_steps: Optional[int]=None,
        render_mode: Optional[str]=None
    ):
        super().__init__()
        self.core = core
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.max_steps = max_steps or (core.rows*core.cols*2)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self._steps = 0

    def _obs(self):
        r, c = self.core.player
        return np.array([
            r/(self.core.rows-1 if self.core.rows>1 else 1),
            c/(self.core.cols-1 if self.core.cols>1 else 1)
        ], dtype=np.float32)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str, Any]]=None):
        super().reset(seed=seed)
        self.core.reset()
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        self._steps += 1
        info_phys = self.core.step_physics(int(action))
        reward = self.reward_fn(info_phys, self._steps)
        terminated = bool(info_phys["reached_goal"])
        truncated = bool(self._steps >= self.max_steps)
        obs = self._obs()
        if self.render_mode == "ansi":
            print(self.core.render_ascii())
            print(f"reward={reward:.3f}, steps={self._steps}, terminated={terminated}, truncated={truncated}")
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "ansi":
            print(self.core.render_ascii())

    def close(self):
        pass


# -----------------------------
#  Entrenamiento / Evaluación
# -----------------------------
ALGOS = {
    "PPO":  ("stable_baselines3", "PPO"),
    "DQN":  ("stable_baselines3", "DQN"),
    "A2C":  ("stable_baselines3", "A2C"),
    "TQC":  ("stable_baselines3", "TQC"),   # si está instalado via extras
    "DDPG": ("stable_baselines3", "DDPG"),
    "SAC":  ("stable_baselines3", "SAC"),
    "TD3":  ("stable_baselines3", "TD3"),
}

def make_env(json_path: Optional[str], reward_fn: Callable[[Dict[str, Any], int], float]) -> MazeEnv:
    core = MazeCore.from_json_file(json_path) if json_path else MazeCore()
    return MazeEnv(core=core, reward_fn=reward_fn, render_mode=None)

def load_algo(algo_name: str):
    algo_name = algo_name.upper()
    if algo_name not in ALGOS:
        raise ValueError(f"Algoritmo no soportado: {algo_name}. Opciones: {list(ALGOS.keys())}")
    pkg, cls = ALGOS[algo_name]
    if pkg != "stable_baselines3":
        raise ValueError("Este ejemplo usa stable-baselines3.")
    from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
    available = {
        "PPO": PPO, "DQN": DQN, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3
    }
    if cls == "TQC":
        try:
            from sb3_contrib import TQC
            available["TQC"] = TQC
        except Exception as e:
            raise RuntimeError("TQC requiere sb3-contrib instalado.") from e
    return available[cls]

def default_model_name(algo: str, reward_id: int, json_path: Optional[str]) -> str:
    yyy = algo.upper()
    dd = f"{int(reward_id):02d}"
    xxx = safe_stem(json_path)
    return f"model_{yyy}_{dd}_{xxx}.zip"

def train(
    algo="PPO",
    steps=200_000,
    json_path=None,
    reward_id: Optional[int]=None,
    save_path: Optional[str]=None
):
    algo = algo.upper()
    # Selección de reward
    rid = reward_id if reward_id is not None else DEFAULT_REWARD_BY_ALGO.get(algo, 1)
    reward_fn = REWARD_FNS.get(rid)
    if reward_fn is None:
        raise ValueError(f"reward_id {rid} no existe. Opciones: {sorted(REWARD_FNS.keys())}")

    env = make_env(json_path, reward_fn)
    Algo = load_algo(algo)

    # Carpeta de logs (TensorBoard)
    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)
    run_name = f"{algo}_r{rid}_{int(time.time())}"
    log_dir = os.path.join(log_root, run_name)

    # hiperparámetros básicos de arranque
    kwargs = {}
    if algo == "DQN":
        kwargs = dict(
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            target_update_interval=500,
            exploration_fraction=0.2,
        )

    model = Algo("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **kwargs)
    model.learn(total_timesteps=int(steps))

    # Nombre automático si no se especifica otro
    final_path = save_path or default_model_name(algo, rid, json_path)
    model.save(final_path)
    env.close()
    print(f"Modelo guardado en: {final_path}")
    print(f"TensorBoard: tensorboard --logdir {log_root}")

def evaluate(model_path, json_path=None, reward_id: Optional[int]=None, episodes=5, render=False):
    # Para evaluar no necesitamos reward_fn (el modelo produce acciones); pero
    # si quieres usar el mismo shaping para logs, lo pasamos. No afecta al modelo entrenado.
    rid = reward_id if reward_id is not None else 1
    reward_fn = REWARD_FNS.get(rid, reward_v1_simple)
    env = make_env(json_path, reward_fn)

    # Cargamos con PPO por defecto; SB3 detecta y error si no coincide.
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
    except Exception:
        # Fallback genérico: intentar con otros algos
        from stable_baselines3 import DQN, A2C, TD3, SAC, DDPG
        tried = []
        for Cls in (DQN, A2C, TD3, SAC, DDPG):
            try:
                model = Cls.load(model_path, env=env)
                break
            except Exception as e:
                tried.append(Cls.__name__)
                model = None
        if model is None:
            raise RuntimeError(f"No pude cargar el modelo automáticamente con PPO ni {tried}.")

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_rew += r
            if render:
                print(env.core.render_ascii())
                print("---")
        rewards.append(ep_rew)
        print(f"Episode {ep+1}: reward={ep_rew:.3f}")
    env.close()
    print(f"Recompensa media: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="PPO, DQN, A2C, SAC, TD3, DDPG, TQC")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--json", type=str, default=None, help="Ruta al laberinto JSON (opcional)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=str, default=None, help="Ruta de salida del modelo (.zip). Si no se da, se usa model_YYY_DD_XXX.zip")
    parser.add_argument("--render", action="store_true", help="Render ASCII en evaluación")
    parser.add_argument("--reward_id", type=int, default=None, help=f"ID de recompensa ({sorted(REWARD_FNS.keys())}); por defecto depende del algoritmo")
    args = parser.parse_args()

    if args.train:
        train(
            algo=args.algo,
            steps=args.steps,
            json_path=args.json,
            reward_id=args.reward_id,
            save_path=args.model
        )
    if args.eval:
        if not args.model:
            # Si eval sin --model, intenta construir nombre por defecto (útil si entrenaste justo antes con esos args)
            rid = args.reward_id if args.reward_id is not None else DEFAULT_REWARD_BY_ALGO.get(args.algo.upper(), 1)
            auto_name = default_model_name(args.algo, rid, args.json)
            print(f"[INFO] --model no dado. Intento cargar: {auto_name}")
            args.model = auto_name
        evaluate(
            model_path=args.model,
            json_path=args.json,
            reward_id=args.reward_id,
            episodes=5,
            render=args.render
        )

if __name__ == "__main__":
    main()
