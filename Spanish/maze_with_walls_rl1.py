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
python maze_with_walls_rl1.py --train --algo PPO --steps 200000

# Entrenar usando tu archivo JSON:
python maze_with_walls_rl1.py --train --algo DQN --steps 300000 --json paths/path_20251104_145523.json

# Elegir una función de recompensa (por ID):
python maze_with_walls_rl1.py --train --algo PPO --reward_id 2 --steps 200000 --json paths/path_20251104_145523.json

# Evaluar el modelo (con render ASCII en consola):
python maze_with_walls_rl1.py --eval --model model_PPO_01_path_20251104_145523.zip --json paths/path_20251104_145523.json --render

# Reproducir modelo sobre un laberinto JSON específico
python viewer_tk.py --model model_DQN_01_path7x7_20251106_154816.zip --json /paths/path7x7_20251106_154816.json
python viewer_tk.py --json /paths/path7x7_20251106_154816.json --model model_PPO_02_path7x7_20251106_154816.zip
python viewer_tk.py --model model_PPO_02_path7x7_20251106_154816.zip --json /paths/path7x7_20251106_154816.json

# O sin JSON (usa filas/cols por defecto)
python viewer_tk.py --model maze_model.zip --rows 12 --cols 16 --cell 36


# Para PPO (recomendado)
python maze_with_walls_rl1.py --train --algo PPO --steps 300000 --json paths/path7x7_20251106_154816.json --reward_id 2

# O para DQN, con recompensa fuerte
python maze_with_walls_rl1.py --train --algo DQN --steps 300000 --json paths/path7x7_20251106_154816.json --reward_id 5
"""

# -----------------------------
#  Utilidades
# -----------------------------
def safe_stem(path: Optional[str]) -> str:
    if not path:
        return "RANDOM"
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
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

        Modificado:
        - Si se sale del tablero o se pisa una celda que NO está en el path,
          se considera un golpe (fell=True), pero el jugador se queda en la
          posición anterior (NO se reinicia al inicio).
        """
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]
        dr, dc = deltas[int(action)]
        prev = self.player
        nr = prev[0] + dr
        nc = prev[1] + dc
        nxt = (nr, nc)

        fell = False
        moved = False
        reached_goal = False

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # Golpe contra borde: no se mueve, pero se marca fell
            fell = True
            self.player = prev
        elif nxt not in self.path_set:
            # Golpe contra espacio vacío: no se mueve, pero se marca fell
            fell = True
            self.player = prev
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
# info = {prev, curr, fell, moved, reached_goal, dist_prev, dist_curr, backtrack?}

FALL_PENALTY = -10.0      # penalización fuerte por golpe (salir del camino/tablero)
BACKTRACK_PENALTY = -5.0  # penalización fuerte por devolverse a la casilla anterior


def reward_v1_simple(info: Dict[str, Any], step_idx: int) -> float:
    """
    Base: +1 al llegar; -10 por caída/golpe; -0.01 por paso;
    penalización adicional por devolverse a la posición anterior.
    """
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return FALL_PENALTY

    reward = -0.01

    # Penalización extra por devolverse
    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


def reward_v2_progress(info: Dict[str, Any], step_idx: int) -> float:
    """
    Shaping por progreso Manhattan hacia la meta, con penalización -10 al caer
    y penalización fuerte por devolverse a la casilla anterior.
    """
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return FALL_PENALTY

    delta = (info["dist_prev"] - info["dist_curr"])  # >0 si se acerca
    reward = 0.02 * delta - 0.01

    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


def reward_v3_sparse(info: Dict[str, Any], step_idx: int) -> float:
    """
    Muy escasa: solo +1 al llegar; leve castigo por tiempo; caída/golpe -10;
    penalización fuerte por devolverse.
    """
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return FALL_PENALTY

    reward = -0.02

    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


def reward_v4_anti_loop(info: Dict[str, Any], step_idx: int) -> float:
    """
    Progreso + anti-bucle suave (penaliza no moverse),
    caída -10 y penalización fuerte por devolverse.
    """
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return FALL_PENALTY

    moved = info["moved"]
    delta = (info["dist_prev"] - info["dist_curr"])
    reward = 0.02 * delta - 0.01
    if not moved:
        reward -= 0.02

    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


def reward_v5_strong(info: Dict[str, Any], step_idx: int) -> float:
    """
    +1 cuando da un paso válido,
    -10 cuando se cae (fuera del camino o del tablero),
    +1000 cuando llega a la meta,
    penalización fuerte por devolverse a la casilla anterior.
    """
    if info["reached_goal"]:
        return 1000.0
    if info["fell"]:
        return FALL_PENALTY

    reward = 1.0

    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


def reward_v6_no_backtrack(info: Dict[str, Any], step_idx: int) -> float:
    """
    Progreso Manhattan + penalización fuerte por 'paso atrás'
    (usando BACKTRACK_PENALTY) + penalización -10 al caer.
    """
    if info["reached_goal"]:
        return 1.0
    if info["fell"]:
        return FALL_PENALTY

    delta = (info["dist_prev"] - info["dist_curr"])  # >0 si se acerca
    reward = 0.02 * delta - 0.01  # shaping por progreso

    # Penalización fuerte por devolverse
    if info.get("backtrack", False):
        reward += BACKTRACK_PENALTY

    return reward


REWARD_FNS: Dict[int, Callable[[Dict[str, Any], int], float]] = {
    1: reward_v1_simple,
    2: reward_v2_progress,
    3: reward_v3_sparse,
    4: reward_v4_anti_loop,
    5: reward_v5_strong,
    6: reward_v6_no_backtrack,
}


DEFAULT_REWARD_BY_ALGO: Dict[str, int] = {
    "PPO": 2,
    "DQN": 1,
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
    Los golpes (caídas) solo dan penalización y se cuentan como fallos,
    pero NO terminan el episodio.
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
        # Máximo de pasos = filas * columnas si no se especifica otro valor
        self.max_steps = max_steps or (self.core.rows * self.core.cols)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self._steps = 0
        self._last_pos = None  # para detectar “paso atrás”
        self._fails = 0        # contador de fallos por episodio

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
        self._last_pos = None 
        self._fails = 0
        return self._obs(), {}

    def step(self, action: int):
        self._steps += 1

        prev_pos_before = self.core.player
        info_phys = self.core.step_physics(int(action))

        backtrack = False
        if info_phys["moved"] and (self._last_pos is not None):
            backtrack = (info_phys["curr"] == self._last_pos)
        info_phys["backtrack"] = backtrack

        # Conteo de fallos
        if info_phys["fell"]:
            self._fails += 1

        # Actualización de _last_pos:
        # si se movió, la casilla anterior es la que tenía antes de mover
        if info_phys["moved"]:
            self._last_pos = prev_pos_before

        reward = self.reward_fn(info_phys, self._steps)

        # FIN DE EPISODIO:
        # - Terminado solo si llegó a la meta.
        # - Truncado si se alcanzó el máximo de pasos.
        terminated = bool(info_phys["reached_goal"])
        truncated = bool(self._steps >= self.max_steps)

        obs = self._obs()

        if self.render_mode == "ansi":
            print(self.core.render_ascii())
            print(
                f"reward={reward:.3f}, steps={self._steps}, "
                f"fails={self._fails}, terminated={terminated}, "
                f"truncated={truncated}, backtrack={backtrack}, fell={info_phys['fell']}"
            )

        info_out = {
            "fails": self._fails,
            "fell": info_phys["fell"],
            "reached_goal": info_phys["reached_goal"],
            "steps": self._steps,
        }

        return obs, reward, terminated, truncated, info_out

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
    "TQC":  ("stable_baselines3", "TQC"),
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
    rid = reward_id if reward_id is not None else DEFAULT_REWARD_BY_ALGO.get(algo, 1)
    reward_fn = REWARD_FNS.get(rid)
    if reward_fn is None:
        raise ValueError(f"reward_id {rid} no existe. Opciones: {sorted(REWARD_FNS.keys())}")

    env = make_env(json_path, reward_fn)
    Algo = load_algo(algo)

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)
    run_name = f"{algo}_r{rid}_{int(time.time())}"
    log_dir = os.path.join(log_root, run_name)

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

    if algo == "PPO":
        model.ent_coef = 0.01
        print("[INFO] Ajuste: PPO con ent_coef=0.01")
    elif algo == "DQN":
        model.exploration_fraction = 0.3
        model.exploration_final_eps = 0.02
        print("[INFO] Ajuste: DQN con exploración 0.3→0.02")

    model.learn(total_timesteps=int(steps))

    final_path = save_path or default_model_name(algo, rid, json_path)
    model.save(final_path)
    env.close()
    print(f"Modelo guardado en: {final_path}")
    print(f"TensorBoard: tensorboard --logdir {log_root}")

def evaluate(model_path, json_path=None, reward_id: Optional[int]=None, episodes=5, render=False):
    rid = reward_id if reward_id is not None else 1
    reward_fn = REWARD_FNS.get(rid, reward_v1_simple)
    env = make_env(json_path, reward_fn)

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
    except Exception:
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
            raise RuntimeError(f"No se pudo cargar el modelo automáticamente con PPO ni {tried}.")

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(action)
            ep_rew += r
            if render:
                print(env.core.render_ascii())
                print(f"fails_en_ep={info.get('fails', 0)}")
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
            rid = args.reward_id if args.reward_id is not None else DEFAULT_REWARD_BY_ALGO.get(args.algo.upper(), 1)
            auto_name = default_model_name(args.algo, rid, args.json)
            print(f"[INFO] --model no dado. Se intenta cargar: {auto_name}")
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
