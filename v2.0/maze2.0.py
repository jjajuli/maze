from time import sleep
import tkinter as tk
from tkinter import messagebox, filedialog
import json
import os
import random
from datetime import datetime, time
from typing import List, Tuple, Set, Dict, Any

# ---------------------------------------------------------
# Tipo auxiliar para representar una celda (fila, columna)
# ---------------------------------------------------------
Cell = Tuple[int, int]


# =========================================================
#               MODO JUEGO: LABERINTO INVISIBLE
# =========================================================
class MazeGame(tk.Tk):
    """
    Ventana principal del juego del laberinto invisible.

    Características principales:
    - El camino correcto es invisible.
    - Si el jugador intenta moverse a una celda que no pertenece al camino
      (o sale del tablero), se cuenta un GOLPE (pared invisible).
    - El jugador NO se reinicia al inicio cuando falla; simplemente no avanza.
    - Se lleva un historial de intentos con:
        * número de intento
        * cantidad de golpes en ese intento
        * cantidad de pasos en ese intento
        * resultado (Meta / Reinicio manual)
    - Se puede configurar el tamaño del tablero n x m desde la interfaz.
    - Se pueden cargar laberintos desde archivos JSON generados por el editor.
    """

    def __init__(self, rows: int = 5, cols: int = 5, cell_size: int = 36):
        """
        Constructor de la ventana del juego.

        Parámetros:
        - rows: número de filas del tablero.
        - cols: número de columnas del tablero.
        - cell_size: tamaño (en píxeles) de cada celda.
        """
        super().__init__()
        self.title("Laberinto Invisible")
        self.resizable(False, False)

        # ---------------- Parámetros del tablero ---------------- #
        # rows / cols: dimensiones del tablero en celdas
        # cell_size: tamaño de cada celda en píxeles
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # margin: margen alrededor del tablero dentro del canvas
        self.margin = 12
        self.canvas_w = self.cols * self.cell_size + self.margin * 2
        self.canvas_h = self.rows * self.cell_size + self.margin * 2

        # ---------------- Estado del laberinto ---------------- #
        # start: celda de inicio
        # goal: celda de meta
        # path: lista ordenada de celdas que forman el camino válido
        # path_set: conjunto para comprobar en O(1) si una celda pertenece al camino
        self.start: Cell = (0, 0)
        self.goal: Cell = (self.rows - 1, self.cols - 1)
        self.path: List[Cell] = []
        self.path_set: Set[Cell] = set()

        # player: posición actual del jugador
        self.player: Cell = self.start

        # loaded_maze: si se carga un laberinto desde archivo, aquí se guarda su definición
        self.loaded_maze: Dict[str, Any] | None = None

        # ---------------- Seguimiento de golpes/intentor ---------------- #
        # current_hits: golpes del intento actual (paredes invisibles)
        # total_hits: golpes acumulados en el laberinto actual
        # current_steps: número de pasos (intentos de movimiento) en el intento actual
        # attempt_index: contador de intentos (1, 2, 3, ...)
        # attempts_history: lista de diccionarios con el resumen de cada intento
        self.current_hits: int = 0
        self.total_hits: int = 0
        self.current_steps: int = 0
        self.attempt_index: int = 1
        self.attempts_history: List[Dict[str, Any]] = []

        # ---------------- Widgets principales ---------------- #
        # Canvas donde se dibuja el tablero
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Diccionario celda -> id de rectángulo del canvas
        self.rect_by_cell: Dict[Cell, int] = {}
        self.build_grid_rects()

        # Frame inferior para controles principales
        self.controls_frame = tk.Frame(self)
        self.controls_frame.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        # Checkbox para mostrar/ocultar visualmente el camino
        self.show_path_var = tk.BooleanVar(value=False)
        self.show_path_cb = tk.Checkbutton(
            self.controls_frame,
            text="Mostrar camino",
            variable=self.show_path_var,
            command=self.render
        )
        self.show_path_cb.grid(row=0, column=0, sticky="w")

        # Botón para generar un nuevo laberinto aleatorio
        self.reset_btn = tk.Button(self.controls_frame, text="Nuevo aleatorio", command=self.new_random_maze)
        self.reset_btn.grid(row=0, column=1, sticky="w", padx=(5, 0))

        # Botón para iniciar un nuevo intento manteniendo el mismo laberinto
        self.retry_btn = tk.Button(self.controls_frame, text="Intentar de nuevo", command=self.on_retry_clicked)
        self.retry_btn.grid(row=0, column=2, sticky="w", padx=(5, 0))

        # Botón para cargar un laberinto desde JSON (creado por el editor)
        self.load_btn = tk.Button(self.controls_frame, text="Cargar laberinto", command=self.load_maze_from_file)
        self.load_btn.grid(row=0, column=3, sticky="w", padx=(5, 0))

        # Etiqueta que muestra los golpes del intento actual
        self.hits_var = tk.StringVar(value="Golpes intento: 0")
        self.hits_lbl = tk.Label(self.controls_frame, textvariable=self.hits_var, font=("Helvetica", 11, "bold"))
        self.hits_lbl.grid(row=0, column=4, sticky="w", padx=(10, 0))

        # Etiqueta que muestra el número de intento y los pasos actuales
        self.attempt_info_var = tk.StringVar(value="Intento 1 | Pasos: 0")
        self.attempt_info_lbl = tk.Label(self.controls_frame, textvariable=self.attempt_info_var)
        self.attempt_info_lbl.grid(row=0, column=5, sticky="w", padx=(10, 0))

        # ---------------- Panel derecho: tamaño, historial y botones extra ---------------- #
        self.side_frame = tk.Frame(self, bd=1, relief="groove")
        self.side_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 10), pady=10)

        # --- Sección tamaño del tablero --- #
        size_title = tk.Label(self.side_frame, text="Tamaño del tablero", font=("Helvetica", 10, "bold"))
        size_title.pack(anchor="w", padx=8, pady=(6, 2))

        size_frame = tk.Frame(self.side_frame)
        size_frame.pack(anchor="w", padx=8, pady=(0, 6))

        tk.Label(size_frame, text="Filas (n):").grid(row=0, column=0, sticky="w")
        tk.Label(size_frame, text="Columnas (m):").grid(row=1, column=0, sticky="w")

        # rows_var / cols_var: variables ligadas a los Spinbox de tamaño
        self.rows_var = tk.IntVar(value=self.rows)
        self.cols_var = tk.IntVar(value=self.cols)

        self.rows_spin = tk.Spinbox(size_frame, from_=2, to=50, width=5, textvariable=self.rows_var)
        self.rows_spin.grid(row=0, column=1, sticky="w", padx=(4, 0))

        self.cols_spin = tk.Spinbox(size_frame, from_=2, to=50, width=5, textvariable=self.cols_var)
        self.cols_spin.grid(row=1, column=1, sticky="w", padx=(4, 0))

        self.apply_size_btn = tk.Button(size_frame, text="Aplicar tamaño", command=self.on_apply_board_size)
        self.apply_size_btn.grid(row=2, column=0, columnspan=2, pady=(4, 0), sticky="we")

        # --- Sección historial de intentos --- #
        history_title = tk.Label(self.side_frame, text="Historial de intentos", font=("Helvetica", 10, "bold"))
        history_title.pack(anchor="w", padx=8, pady=(8, 2))

        # Listbox donde se muestra el resumen de cada intento con capacidad de scroll con desplazamiento automático 
        self.history_scrollbar = tk.Scrollbar(self.side_frame, orient="vertical")
        self.history_scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=(0, 6))
        self.history_listbox = tk.Listbox(self.side_frame, width=36, height=14, yscrollcommand=self.history_scrollbar.set)
        self.history_listbox.pack(anchor="w", padx=8, pady=(0, 6))
        self.history_scrollbar.config(command=self.history_listbox.yview)

        # Botón para limpiar el historial sin cambiar el laberinto
        self.clear_history_btn = tk.Button(self.side_frame, text="Limpiar historial", command=self.clear_history)
        self.clear_history_btn.pack(anchor="w", padx=8, pady=(0, 8))

        # --- Botones extra (Boton1 ... Boton6) --- #
        extra_title = tk.Label(self.side_frame, text="Botones extra", font=("Helvetica", 10, "bold"))
        extra_title.pack(anchor="w", padx=8, pady=(4, 2))

        extra_btns_frame = tk.Frame(self.side_frame)
        extra_btns_frame.pack(anchor="w", padx=8, pady=(0, 8))

        # Boton1..Boton6 preparados para futuras funcionalidades
        self.btn1 = tk.Button(extra_btns_frame, text="Boton1", width=10, command=self.on_boton1)
        self.btn2 = tk.Button(extra_btns_frame, text="Boton2", width=10, command=self.on_boton2)
        self.btn3 = tk.Button(extra_btns_frame, text="Boton3", width=10, command=self.on_boton3)
        self.btn4 = tk.Button(extra_btns_frame, text="Boton4", width=10, command=self.on_boton4)
        self.btn5 = tk.Button(extra_btns_frame, text="Boton5", width=10, command=self.on_boton5)
        self.btn6 = tk.Button(extra_btns_frame, text="Boton6", width=10, command=self.on_boton6)

        # Distribución en dos columnas (3 filas x 2 columnas)
        self.btn1.grid(row=0, column=0, padx=2, pady=2, sticky="we")
        self.btn2.grid(row=0, column=1, padx=2, pady=2, sticky="we")
        self.btn3.grid(row=1, column=0, padx=2, pady=2, sticky="we")
        self.btn4.grid(row=1, column=1, padx=2, pady=2, sticky="we")
        self.btn5.grid(row=2, column=0, padx=2, pady=2, sticky="we")
        self.btn6.grid(row=2, column=1, padx=2, pady=2, sticky="we")

        # ---------------- Controles de teclado ---------------- #
        # Se capturan flechas y teclas WASD para mover al jugador
        self.bind_all("<Up>", lambda e: self.try_move((-1, 0)))
        self.bind_all("<Down>", lambda e: self.try_move((1, 0)))
        self.bind_all("<Left>", lambda e: self.try_move((0, -1)))
        self.bind_all("<Right>", lambda e: self.try_move((0, 1)))
        self.bind_all("w", lambda e: self.try_move((-1, 0)))
        self.bind_all("s", lambda e: self.try_move((1, 0)))
        self.bind_all("a", lambda e: self.try_move((0, -1)))
        self.bind_all("d", lambda e: self.try_move((0, 1)))

        # ---------------- Inicialización del laberinto ---------------- #
        # Se genera un laberinto aleatorio y se prepara el primer intento
        self.generate_path_random()
        self.reset_attempts_for_new_maze()
        self.render()

        self.update_idletasks()

    # -----------------------------------------------------
    # Construcción de la grilla (rectángulos en el canvas)
    # -----------------------------------------------------
    def build_grid_rects(self):
        """
        Crea o recrea los rectángulos del canvas según filas, columnas y tamaño.

        Variables internas:
        - self.canvas_w / self.canvas_h: dimensiones del canvas en píxeles.
        - self.rect_by_cell: diccionario que mapea cada celda (r, c) al id
          del rectángulo correspondiente en el canvas.
        """
        # Ajustar dimensiones del canvas
        self.canvas_w = self.cols * self.cell_size + self.margin * 2
        self.canvas_h = self.rows * self.cell_size + self.margin * 2
        self.canvas.config(width=self.canvas_w, height=self.canvas_h)

        # Limpiar cualquier rectángulo previo
        self.canvas.delete("all")
        self.rect_by_cell.clear()

        # Crear la rejilla de celdas
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.margin + c * self.cell_size
                y0 = self.margin + r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill="#000",
                    outline="#222"
                )
                self.rect_by_cell[(r, c)] = rect

    # -----------------------------------------------------
    # Cambio de tamaño del tablero (n * m) desde la UI
    # -----------------------------------------------------
    def apply_board_size(self, rows: int, cols: int, cell_size: int | None = None):
        """
        Cambia las dimensiones del tablero y reconstruye la rejilla.

        Parámetros:
        - rows: nuevas filas.
        - cols: nuevas columnas.
        - cell_size: tamaño de celda opcional, si se quiere modificar también.
        """
        self.rows = int(rows)
        self.cols = int(cols)
        if cell_size is not None:
            self.cell_size = int(cell_size)

        # Reiniciar posiciones de inicio/meta/jugador con las nuevas dimensiones
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.player = self.start

        # Reconstruir la rejilla gráfica
        self.build_grid_rects()

    def on_apply_board_size(self):
        """
        Lee los valores de filas y columnas de los controles de la derecha
        y aplica el nuevo tamaño, regenerando un laberinto y reseteando los intentos.
        """
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get())
            if new_rows < 2 or new_cols < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Valor inválido", "Filas y columnas deben ser enteros mayor o igual a 2.")
            return

        # Aplicar el nuevo tamaño y generar un nuevo camino
        self.apply_board_size(new_rows, new_cols, self.cell_size)
        self.generate_path_random()
        self.reset_attempts_for_new_maze()
        self.render()

    # -----------------------------------------------------
    # Generación de laberintos aleatorios (camino único)
    # -----------------------------------------------------
    def new_random_maze(self):
        """
        Genera un nuevo camino aleatorio y olvida cualquier laberinto cargado.
        También resetea el historial de intentos para el nuevo laberinto.
        """
        self.loaded_maze = None
        self.generate_path_random()
        self.reset_attempts_for_new_maze()
        self.render()

    def generate_path_random(self):
        """
        Genera un camino válido entre inicio y meta usando una DFS aleatoria.

        Variables internas:
        - dirs: posibles movimientos (arriba, abajo, izquierda, derecha).
        - visited: conjunto de celdas ya visitadas.
        - parent: diccionario para reconstruir el camino padre -> hijo.
        """
        self.path = self._dfs_path(self.start, self.goal)
        tries = 0
        # Si por alguna razón no se encuentra camino, se intenta varias veces
        while not self.path and tries < 10:
            tries += 1
            self.path = self._dfs_path(self.start, self.goal)

        if not self.path:
            # Fallback: se construye un camino en forma de "L"
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
        """
        Implementa una búsqueda en profundidad aleatoria (DFS) para encontrar un camino.

        Parámetros:
        - start: celda de inicio.
        - goal: celda objetivo.

        Retorna:
        - Lista de celdas que conforman el camino desde start hasta goal.
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = {start}
        parent: Dict[Cell, Cell] = {}
        stack: List[Cell] = [start]

        while stack:
            cur = stack.pop()
            if cur == goal:
                break
            r, c = cur
            random.shuffle(dirs)  # desordenar direcciones para variar laberinto
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols and nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    stack.append(nxt)

        if goal not in parent and goal != start:
            return []

        # Reconstruir el camino desde goal hacia start usando 'parent'
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # -----------------------------------------------------
    # Carga de laberintos desde archivo JSON
    # -----------------------------------------------------
    def load_maze_from_file(self):
        """
        Permite seleccionar un archivo JSON con un laberinto y lo aplica.
        El formato es compatible con el editor de caminos.
        """
        filepath = filedialog.askopenfilename(
            title="Selecciona un archivo de laberinto",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=os.path.abspath("paths") if os.path.isdir("paths") else os.getcwd()
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_loaded_maze(data)
            messagebox.showinfo("Laberinto cargado", f"Se cargó el laberinto desde:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")

    def apply_loaded_maze(self, data: Dict[str, Any]):
        """
        Aplica dimensiones, inicio/meta y camino desde un diccionario cargado de JSON.

        Claves esperadas en 'data':
        - rows, cols, cell_size: tamaño del tablero.
        - start, goal: pares [fila, columna].
        - path: lista de pares [fila, columna] que definen el camino.
        """
        rows = int(data.get("rows", self.rows))
        cols = int(data.get("cols", self.cols))
        cell_size = int(data.get("cell_size", self.cell_size))
        start_list = data.get("start", [0, 0])
        goal_list = data.get("goal", [rows - 1, cols - 1])
        path_list = data.get("path", [])

        # Ajustar tablero a las dimensiones del archivo
        self.apply_board_size(rows, cols, cell_size)

        # Establecer inicio y meta (acotando por seguridad)
        sr, sc = int(start_list[0]), int(start_list[1])
        gr, gc = int(goal_list[0]), int(goal_list[1])
        sr = max(0, min(self.rows - 1, sr))
        sc = max(0, min(self.cols - 1, sc))
        gr = max(0, min(self.rows - 1, gr))
        gc = max(0, min(self.cols - 1, gc))
        self.start = (sr, sc)
        self.goal = (gr, gc)
        self.player = self.start

        # Construir conjunto de celdas del camino
        path: List[Cell] = []
        for rc in path_list:
            if isinstance(rc, list) and len(rc) == 2:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    path.append((r, c))

        # Asegurar que inicio y meta estén en el camino
        if self.start not in path:
            path.insert(0, self.start)
        if self.goal not in path:
            path.append(self.goal)

        self.path = path
        self.path_set = set(self.path)

        # Guardar para poder reiniciar el mismo laberinto tras ganar
        self.loaded_maze = {
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
            "start": self.start,
            "goal": self.goal,
            "path": [list(c) for c in self.path],
        }

        # Al cargar un nuevo laberinto se resetean intentos e historial
        self.reset_attempts_for_new_maze()
        self.render()

    # -----------------------------------------------------
    # Lógica de movimiento con PAREDES INVISIBLES
    # -----------------------------------------------------
    def try_move(self, delta: Tuple[int, int]):
        """
        Intenta mover al jugador según delta (dr, dc).

        Parámetros:
        - delta: tupla (dr, dc) con el desplazamiento en filas y columnas.

        Comportamiento:
        - Si la celda destino está fuera del tablero, se cuenta un golpe y el
          jugador no se mueve.
        - Si la celda destino no está en el camino válido, también se cuenta
          un golpe y el jugador no se mueve.
        - Si la celda destino está en el camino, el jugador se mueve.
        - Al llegar a la meta, se considera que el intento termina con éxito.
        """
        # Cada pulsación de movimiento cuenta como un paso en el intento
        self.current_steps += 1
        self.update_attempt_info_label()

        nr = self.player[0] + delta[0]
        nc = self.player[1] + delta[1]
        nxt = (nr, nc)

        # Caso 1: fuera de límites -> golpe (pared invisible externa)
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            self.register_hit()
            # Se puede parpadear la celda actual para indicar el choque
            self.flash_cell(self.player, color="#a00")
            return

        # Caso 2: dentro del tablero pero fuera del camino -> golpe (pared invisible)
        if nxt not in self.path_set:
            self.register_hit()
            self.flash_cell(self.player, color="#a00")
            return

        # Caso 3: movimiento válido sobre el camino
        self.player = nxt
        self.render()

        # Comprobar si se llegó a la meta
        if self.player == self.goal:
            self.on_win()

    def register_hit(self):
        """
        Registra un golpe contra una pared invisible en el intento actual.

        Efectos:
        - Incrementa current_hits y total_hits.
        - Actualiza la etiqueta de golpes.
        """
        self.current_hits += 1
        self.total_hits += 1
        self.update_hits_label()

    def on_retry_clicked(self):
        """
        Callback del botón 'Intentar de nuevo'.

        Comportamiento:
        - Cierra el intento actual (resultado: 'Reinicio').
        - Registra ese intento en el historial.
        - Crea un nuevo intento desde el inicio con contadores a cero.
        """
        self.end_current_attempt(completed=False)
        self.start_new_attempt()

    def on_win(self):
        """
        Llamado cuando el jugador alcanza la meta.

        Comportamiento:
        - Cierra el intento actual (resultado: 'Meta').
        - Muestra un mensaje con el resumen de golpes.
        - Vuelve a cargar el mismo laberinto si venía de un archivo,
          o genera uno aleatorio nuevo.
        - Inicia un nuevo intento para el laberinto recién preparado.
        """
        # Finalizar intento actual como exitoso
        self.end_current_attempt(completed=True)

        # Resumen del intento que acaba de terminar
        last = self.attempts_history[-1] if self.attempts_history else None
        golpes = last["golpes"] if last else self.current_hits
        pasos = last["pasos"] if last else self.current_steps

        messagebox.showinfo(
            "¡Llegaste!",
            f"Has alcanzado la meta.\n"
            f"Golpes en este intento: {golpes}\n"
            f"Pasos en este intento: {pasos}"
        )

        # Reiniciar manteniendo el mismo laberinto si fue cargado,
        # o generando uno nuevo en caso contrario.
        if self.loaded_maze is not None:
            self.apply_loaded_maze(self.loaded_maze)
        else:
            self.new_random_maze()

    # -----------------------------------------------------
    # Gestión de intentos e historial
    # -----------------------------------------------------
    def reset_attempts_for_new_maze(self):
        """
        Prepara el estado interno cuando se comienza a usar un nuevo laberinto.

        Efectos:
        - Limpia el historial de intentos.
        - Reinicia contadores de golpes totales.
        - Fija el índice de intento en 1.
        - Inicia el primer intento.
        """
        self.attempts_history.clear()
        self.history_listbox.delete(0, tk.END)
        self.total_hits = 0
        self.attempt_index = 1
        self.start_new_attempt()

    def start_new_attempt(self):
        """
        Inicia un nuevo intento desde la celda de inicio del laberinto.

        Efectos:
        - Coloca al jugador en start.
        - Resetea current_hits y current_steps.
        - Actualiza etiquetas de información.
        """
        self.player = self.start
        self.current_hits = 0
        self.current_steps = 0
        self.update_hits_label()
        self.update_attempt_info_label()
        self.render()

    def end_current_attempt(self, completed: bool):
        """
        Cierra el intento actual y lo añade al historial.

        Parámetros:
        - completed: True si el intento termina por llegar a la meta,
                     False si termina por reinicio manual.

        Efectos:
        - Inserta un registro en attempts_history.
        - Añade una línea descriptiva en el Listbox de historial.
        - Incrementa el índice de intento para el siguiente.
        """
        resultado = "Meta" if completed else "Reinicio"
        record = {
            "numero": self.attempt_index,
            "golpes": self.current_hits,
            "pasos": self.current_steps,
            "resultado": resultado,
        }
        self.attempts_history.append(record)

        # Texto legible que se muestra en el historial
        line = (
            f"Intento {record['numero']}: "
            f"golpes={record['golpes']}, "
            f"pasos={record['pasos']}, "
            f"resultado={record['resultado']}"
        )
        self.history_listbox.insert(tk.END, line)
        self.attempt_index += 1

    def clear_history(self):
        """
        Limpia el historial de intentos sin modificar el laberinto ni el intento actual.
        """
        self.attempts_history.clear()
        self.history_listbox.delete(0, tk.END)

    def update_hits_label(self):
        """
        Actualiza el texto de la etiqueta de golpes del intento actual.
        """
        self.hits_var.set(f"Golpes intento: {self.current_hits}")

    def update_attempt_info_label(self):
        """
        Actualiza el texto de la etiqueta que muestra el número de intento
        y los pasos actuales.
        """
        self.attempt_info_var.set(f"Intento {self.attempt_index} | Pasos: {self.current_steps}")

    # -----------------------------------------------------
    # Renderizado del tablero
    # -----------------------------------------------------
    def render(self):
        """
        Redibuja las celdas del tablero en el canvas.

        Comportamiento:
        - Si 'Mostrar camino' está activado, las celdas del camino se muestran en gris.
        - La celda de inicio es verde, la de meta es amarilla.
        - La celda del jugador se muestra en azul.
        """
        show_path = self.show_path_var.get()
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                rect = self.rect_by_cell.get(cell)
                if rect is None:
                    continue

                # color por defecto: negro
                fill = "#000"
                outline = "#222"

                # Mostrar camino si está activado
                if show_path and cell in self.path_set:
                    fill = "#2b2b2b"  # camino en gris

                # Inicio y meta
                if cell == self.start:
                    fill = "#2ecc71"  # verde
                if cell == self.goal:
                    fill = "#f1c40f"  # amarillo

                # Jugador por encima de todo
                if cell == self.player:
                    fill = "#3498db"  # azul
                    outline = "#5faee3"

                self.canvas.itemconfig(rect, fill=fill, outline=outline)

        self.canvas.update_idletasks()

    def flash_cell(self, cell: Cell, color: str = "#a00", flashes: int = 1, delay_ms: int = 60):
        """
        Hace parpadear una celda del tablero con un color dado para indicar un evento
        (por ejemplo, un golpe contra una pared invisible).

        Parámetros:
        - cell: celda que se quiere resaltar.
        - color: color temporal del parpadeo.
        - flashes: cuántas veces parpadea.
        - delay_ms: retraso en milisegundos entre cambios de color.
        """
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

    # -----------------------------------------------------
    # Botones extra (Boton1 ... Boton6) - stubs
    # -----------------------------------------------------
    def on_boton1(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton1.
        """
        # loop infinito de movimientos aleatorios como ejemplo
        while True:
            movement = random.randint(0, 3)
            match movement:
                case 0:  # Up
                    self.try_move((-1, 0))
                    self.history_listbox.insert(tk.END, "UP")
                case 1:  # Right
                    self.try_move((0, 1))
                    self.history_listbox.insert(tk.END, "RIGHT")
                case 2:  # Down
                    self.try_move((1, 0))
                    self.history_listbox.insert(tk.END, "DOWN")
                case 3:  # Left
                    self.try_move((0, -1))
                    self.history_listbox.insert(tk.END, "LEFT")
            
            self.history_listbox.see(tk.END)
            # pausa para ver los movimientos    
            self.update()
            self.after(200) 
            # Chequeo si llegó a la meta para salir del loop
            if self.player == self.goal:
                print("Llegó a la meta")
                break
            

        #messagebox.showinfo("Boton1", "Espacio reservado para una función futura (Boton1).")

    def on_boton2(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton2.
        """
        messagebox.showinfo("Boton2", "Espacio reservado para una función futura (Boton2).")

    def on_boton3(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton3.
        """
        messagebox.showinfo("Boton3", "Espacio reservado para una función futura (Boton3).")

    def on_boton4(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton4.
        """
        messagebox.showinfo("Boton4", "Espacio reservado para una función futura (Boton4).")

    def on_boton5(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton5.
        """
        messagebox.showinfo("Boton5", "Espacio reservado para una función futura (Boton5).")

    def on_boton6(self):
        """
        Espacio reservado para la funcionalidad futura asociada a Boton6.
        """
        messagebox.showinfo("Boton6", "Espacio reservado para una función futura (Boton6).")


# =========================================================
#           MODO EDITOR: CREACIÓN DE CAMINOS JSON
# =========================================================
class PathEditorGame(tk.Tk):
    """
    Versión "editor": el usuario define el camino moviéndose.

    Funcionalidad:
    - La estela recorrida queda pintada en gris.
    - Inicio (verde) y Meta (amarillo) siempre visibles.
    - Al llegar a la meta se puede guardar el camino en un JSON.
    - Los archivos generados son compatibles con MazeGame.load_maze_from_file.
    """

    def __init__(self, rows: int = 7, cols: int = 7, cell_size: int = 36):
        super().__init__()
        self.title("Editor de Caminos - Laberinto Invisible")
        self.resizable(False, False)

        # --- Configuración de la grilla ---
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.margin = 12
        self.canvas_w = self.cols * self.cell_size + self.margin * 2
        self.canvas_h = self.rows * self.cell_size + self.margin * 2

        # --- Estado del editor ---
        self.start: Cell = (0, 0)
        self.goal: Cell = (self.rows - 1, self.cols - 1)
        self.player: Cell = self.start

        # trail: secuencia de celdas visitadas por el jugador
        self.trail: List[Cell] = [self.start]
        self.trail_set: Set[Cell] = {self.start}

        # --- UI principal ---
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        self.new_btn = tk.Button(self, text="Nuevo (limpiar)", command=self.reset_board)
        self.new_btn.grid(row=1, column=0, sticky="w", padx=(12, 0))

        self.restart_btn = tk.Button(self, text="Volver al inicio", command=self.restart_player)
        self.restart_btn.grid(row=1, column=1, sticky="w")

        self.save_btn = tk.Button(self, text="Guardar camino", command=self.save_path)
        self.save_btn.grid(row=1, column=2, sticky="w")

        self.load_btn = tk.Button(self, text="Cargar y dibujar camino", command=self.load_and_draw_path)
        self.load_btn.grid(row=1, column=3, sticky="w")

        self.info_lbl = tk.Label(self, text="Mueve con flechas/WASD. Llega a la meta para guardar.")
        self.info_lbl.grid(row=1, column=4, sticky="e", padx=(0, 12))

        # Rectángulos por celda
        self.rect_by_cell: Dict[Cell, int] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.margin + c * self.cell_size
                y0 = self.margin + r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#000", outline="#222")
                self.rect_by_cell[(r, c)] = rect

        # Controles de teclado (flechas + WASD)
        self.bind_all("<Up>",   lambda e: self.try_move((-1, 0)))
        self.bind_all("<Down>", lambda e: self.try_move((1, 0)))
        self.bind_all("<Left>", lambda e: self.try_move((0, -1)))
        self.bind_all("<Right>", lambda e: self.try_move((0, 1)))
        self.bind_all("w", lambda e: self.try_move((-1, 0)))
        self.bind_all("s", lambda e: self.try_move((1, 0)))
        self.bind_all("a", lambda e: self.try_move((0, -1)))
        self.bind_all("d", lambda e: self.try_move((0, 1)))

        self.render()

    # ----------------- Movimiento en el editor ----------------- #
    def try_move(self, delta: Tuple[int, int]):
        """
        Mueve al jugador dentro de los límites del tablero y registra la estela.

        Parámetros:
        - delta: tupla (dr, dc) con el desplazamiento.
        """
        nr = self.player[0] + delta[0]
        nc = self.player[1] + delta[1]
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # Fuera del tablero: se ignora el movimiento
            return
        nxt = (nr, nc)
        self.player = nxt

        # Registrar la celda en el trail (permitiendo repeticiones en la secuencia)
        self.trail.append(nxt)
        self.trail_set.add(nxt)
        self.render()

        if self.player == self.goal:
            self.on_reach_goal()

    def on_reach_goal(self):
        """
        Pregunta si se desea guardar el camino al llegar a la meta.
        """
        ans = messagebox.askyesno("Meta alcanzada", "¿Quieres guardar este camino con timestamp?")
        if ans:
            self.save_path()

    # ----------------- Guardado / Carga ----------------- #
    def build_path_payload(self) -> Dict[str, Any]:
        """
        Construye el diccionario que se guardará como JSON.

        Claves:
        - rows, cols, cell_size
        - start, goal
        - path: lista de [fila, columna] visitadas (recorrido completo)
        - created_at: marca de tiempo ISO-8601.
        """
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
        """
        Guarda el camino actual en un archivo JSON dentro de ./paths.
        """
        os.makedirs("paths", exist_ok=True)
        payload = self.build_path_payload()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = f"path_{timestamp}.json"
        filepath = filedialog.asksaveasfilename(
            title="Guardar camino",
            defaultextension=".json",
            initialdir=os.path.abspath("paths"),
            initialfile=default_name,
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Guardado", f"Camino guardado en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{e}")

    def load_and_draw_path(self):
        """
        Carga un archivo JSON con camino y lo dibuja en el tablero actual.
        """
        filepath = filedialog.askopenfilename(
            title="Cargar camino",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=os.path.abspath("paths") if os.path.isdir("paths") else os.getcwd()
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")
            return

        # Ajustar dimensiones si el archivo trae filas/columnas diferentes
        rows = int(data.get("rows", self.rows))
        cols = int(data.get("cols", self.cols))
        cell_size = int(data.get("cell_size", self.cell_size))

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.canvas_w = self.cols * self.cell_size + self.margin * 2
        self.canvas_h = self.rows * self.cell_size + self.margin * 2
        self.canvas.config(width=self.canvas_w, height=self.canvas_h)

        # Reconstruir la rejilla
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

        # Aplicar start, goal y trail
        start_list = data.get("start", [0, 0])
        goal_list = data.get("goal", [self.rows - 1, self.cols - 1])
        path_list = data.get("path", [])

        self.start = (int(start_list[0]), int(start_list[1]))
        self.goal = (int(goal_list[0]), int(goal_list[1]))
        self.player = self.start

        self.trail = []
        self.trail_set = set()
        for rc in path_list:
            if isinstance(rc, list) and len(rc) == 2:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    cell = (r, c)
                    self.trail.append(cell)
                    self.trail_set.add(cell)

        if not self.trail:
            self.trail = [self.start]
            self.trail_set = {self.start}

        self.render()
        messagebox.showinfo("Camino cargado", f"Se cargó el camino desde:\n{filepath}")

    # ----------------- Reseteo y renderizado ----------------- #
    def reset_board(self):
        """
        Limpia el tablero y reinicia el recorrido al estado inicial.
        """
        self.player = self.start
        self.trail = [self.start]
        self.trail_set = {self.start}
        self.render()

    def restart_player(self):
        """
        Mueve al jugador a la celda de inicio sin borrar la estela.
        """
        self.player = self.start
        self.render()

    def render(self):
        """
        Redibuja la grilla del editor, mostrando:
        - Estela gris donde se ha pasado.
        - Inicio verde.
        - Meta amarilla.
        - Jugador azul.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                rect = self.rect_by_cell.get(cell)
                if rect is None:
                    continue

                fill = "#000"
                outline = "#222"

                if cell in self.trail_set:
                    fill = "#2b2b2b"  # trail en gris

                if cell == self.start:
                    fill = "#2ecc71"  # inicio verde
                if cell == self.goal:
                    fill = "#f1c40f"  # meta amarilla

                if cell == self.player:
                    fill = "#3498db"  # jugador azul
                    outline = "#5faee3"

                self.canvas.itemconfig(rect, fill=fill, outline=outline)

        self.canvas.update_idletasks()


# =========================================================
#                        PUNTO DE ENTRADA
# =========================================================
def main():
    """
    Punto de entrada principal.

    Por defecto se lanza el juego del laberinto invisible (MazeGame).
    Si se desea lanzar el editor de caminos, se puede comentar la parte
    de MazeGame y descomentar la de PathEditorGame.
    """
    # Modo juego (solicitudes del usuario aplicadas aquí):
    app = MazeGame(rows=5, cols=5, cell_size=36)
    app.mainloop()

    # Modo editor (opcional, si se quiere ejecutar el editor por separado):
    # editor = PathEditorGame(rows=7, cols=7, cell_size=36)
    # editor.mainloop()


if __name__ == "__main__":
    main()
