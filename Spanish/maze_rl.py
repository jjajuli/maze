import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import argparse
import os

Cell = Tuple[int, int]


'''
Cómo usarlo

# Entrenar con un laberinto aleatorio (sin Tk, rápido):

python maze_rl.py --train --algo PPO --steps 200000

# Entrenar usando tu archivo JSON:

python maze_rl.py --train --algo DQN --steps 300000 --json paths/path_20251104_145523.json
python maze_rl.py --train --algo PPO --steps 300000 --json paths/path_20251104_145523.json
python maze_rl.py --train --algo PPO --steps 300000 --json paths/path_20251104_145523.json

# Evaluar el modelo (con render ASCII en consola):

python maze_rl.py --eval --model maze_model.zip --json paths/path_20251104_145523 --render

'''




# -----------------------------
#  MazeCore: lógica sin GUI
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

    def reset(self):
        self.player = self.start
        return self.player

    def step(self, action: int):
        # acciones: 0=up,1=down,2=left,3=right
        drdc = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nr = self.player[0] + drdc[0]
        nc = self.player[1] + drdc[1]
        nxt = (nr, nc)

        fell = False
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            fell = True
        elif nxt not in self.path_set:
            fell = True

        if fell:
            # “caída” -> castigo y volver al inicio
            self.player = self.start
            reward = -0.1
            terminated = False
        else:
            self.player = nxt
            if self.player == self.goal:
                reward = +1.0
                terminated = True
            else:
                reward = -0.01  # shaping para incentivar trayectos cortos
                terminated = False
        return self.player, reward, terminated

    def _default_L_path(self) -> List[Cell]:
        # Camino simple en L, por si no hay JSON
        path = []
        r, c = self.start
        while c < self.goal[1]:
            path.append((r,c)); c += 1
        while r < self.goal[0]:
            path.append((r,c)); r += 1
        path.append(self.goal)
        return path

    # Render ASCII opcional (para depurar/evaluar sin Tk)
    def render_ascii(self) -> str:
        grid = [[" ." for _ in range(self.cols)] for _ in range(self.rows)]
        for (r,c) in self.path_set:
            grid[r][c] = "  "  # camino
        sr, sc = self.start; grid[sr][sc] = " S"
        gr, gc = self.goal;  grid[gr][gc] = " G"
        pr, pc = self.player; grid[pr][pc] = " P"
        return "\n".join("".join(row) for row in grid)


# -----------------------------
#  Gymnasium Env
# -----------------------------
class MazeEnv(gym.Env):
    """
    Observación: (fila, col) como Box float32 normalizada [0,1], shape=(2,)
    Acciones: Discrete(4) -> up,down,left,right
    Termina al llegar a la meta o por límite de pasos.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, core: MazeCore, max_steps: Optional[int]=None, render_mode: Optional[str]=None):
        super().__init__()
        self.core = core
        self.render_mode = render_mode
        self.max_steps = max_steps or (core.rows*core.cols*2)

        # espacios
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._steps = 0

    def _obs(self):
        r, c = self.core.player
        return np.array([r/(self.core.rows-1 if self.core.rows>1 else 1),
                         c/(self.core.cols-1 if self.core.cols>1 else 1)], dtype=np.float32)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str, Any]]=None):
        super().reset(seed=seed)
        self.core.reset()
        self._steps = 0
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self._steps += 1
        _, reward, terminated = self.core.step(int(action))
        truncated = self._steps >= self.max_steps
        obs = self._obs()
        info = {}
        if self.render_mode == "ansi":
            print(self.core.render_ascii())
            print(f"reward={reward:.3f}, steps={self._steps}, terminated={terminated}, truncated={truncated}")
        return obs, reward, terminated, truncated, info

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

def make_env(json_path: Optional[str]) -> MazeEnv:
    core = MazeCore.from_json_file(json_path) if json_path else MazeCore()
    return MazeEnv(core=core, render_mode=None)

def load_algo(algo_name: str):
    algo_name = algo_name.upper()
    if algo_name not in ALGOS:
        raise ValueError(f"Algoritmo no soportado: {algo_name}. Opciones: {list(ALGOS.keys())}")
    pkg, cls = ALGOS[algo_name]
    if pkg != "stable_baselines3":
        raise ValueError("Este ejemplo usa stable-baselines3.")
    from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
    # TQC está en sb3-contrib; si lo usas, impórtalo allí.
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

def train(algo="PPO", steps=200_000, json_path=None, save_path="maze_model.zip"):
    env = make_env(json_path)
    Algo = load_algo(algo)
    # Carpeta de logs
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    # hiperparámetros sensatos de arranque para discreto
    kwargs = {}
    if algo.upper() == "DQN":
        kwargs = dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, target_update_interval=500)
    model = Algo("MlpPolicy", env, verbose=1,  tensorboard_log=log_dir, **kwargs)
    model.learn(total_timesteps=int(steps))
    model.save(save_path)
    env.close()
    print(f"Modelo guardado en: {save_path}")

def evaluate(model_path, json_path=None, episodes=5, render=False):
    from stable_baselines3.common.base_class import BaseAlgorithm
    env = make_env(json_path)
    # detecta clase por cabecera del zip
    # para simplicidad cargamos con PPO; si fue DQN, SB3 lo detecta por el zip
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
    except Exception:
        # fallback genérico
        from stable_baselines3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(model_path)
        algo_name = data.get("policy_class", "Unknown")
        raise RuntimeError(f"No pude cargar el modelo automáticamente. Algoritmo detectado: {algo_name}")

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
    parser.add_argument("--model", type=str, default="maze_model.zip")
    parser.add_argument("--render", action="store_true", help="Render ASCII en evaluación")
    args = parser.parse_args()

    if args.train:
        train(algo=args.algo, steps=args.steps, json_path=args.json, save_path=args.model)
    if args.eval:
        evaluate(model_path=args.model, json_path=args.json, episodes=5, render=args.render)

if __name__ == "__main__":
    main()
