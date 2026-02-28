import tkinter as tk
from tkinter import messagebox
import heapq, random, math, time

CELL = 30
EMPTY, WALL, START, GOAL = 0, 1, 2, 3

COLORS = {
    EMPTY: "white", WALL: "black", START: "green", GOAL: "red",
    "visited": "lightblue", "frontier": "yellow", "path": "orange", "agent": "purple"
}

# ---------- Heuristics ----------
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ---------- Neighbors ----------
def neighbors(grid, rows, cols, pos):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

# ---------- A* ----------
def astar(grid, rows, cols, start, goal, h):
    open_list = [(h(start,goal), 0, start)]
    came_from, g_cost = {}, {start: 0}
    visited, frontier = set(), set([start])

    while open_list:
        _, g, cur = heapq.heappop(open_list)
        frontier.discard(cur)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            return rebuild_path(came_from, start, goal), visited
        for nb in neighbors(grid, rows, cols, cur):
            new_g = g_cost[cur] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                heapq.heappush(open_list, (new_g + h(nb,goal), new_g, nb))
                came_from[nb] = cur
                frontier.add(nb)
    return None, visited

# ---------- GBFS ----------
def gbfs(grid, rows, cols, start, goal, h):
    open_list = [(h(start,goal), start)]
    came_from, visited, seen = {}, set(), set([start])
    frontier = set([start])

    while open_list:
        _, cur = heapq.heappop(open_list)
        frontier.discard(cur)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            return rebuild_path(came_from, start, goal), visited
        for nb in neighbors(grid, rows, cols, cur):
            if nb not in seen:
                seen.add(nb)
                heapq.heappush(open_list, (h(nb,goal), nb))
                came_from[nb] = cur
                frontier.add(nb)
    return None, visited

# ---------- Rebuild path ----------
def rebuild_path(came_from, start, goal):
    path, node = [], goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

# ---------- Run search ----------
def search(grid, rows, cols, start, goal, algo, heuristic):
    h = manhattan if heuristic == "Manhattan" else euclidean
    if algo == "A*":
        return astar(grid, rows, cols, start, goal, h)
    return gbfs(grid, rows, cols, start, goal, h)

# ============================================================
# Main Application
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Visualizer")
        self.rows, self.cols = 15, 20
        self.start, self.goal = (0,0), (14,19)
        self.grid = self.empty_grid()

        # animation state
        self.running = False
        self.path, self.path_idx, self.visited = [], 0, set()
        self.agent, self.after_id = None, None

        self.build_panel()
        self.build_canvas()
        self.draw()

    def empty_grid(self):
        g = [[EMPTY]*self.cols for _ in range(self.rows)]
        g[self.start[0]][self.start[1]] = START
        g[self.goal[0]][self.goal[1]]   = GOAL
        return g

    # ---------- Control Panel ----------
    def build_panel(self):
        p = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        p.pack(side=tk.LEFT, fill=tk.Y)

        def label(text):
            tk.Label(p, text=text, font=("Arial",10,"bold"), bg="#f0f0f0").pack(anchor="w", pady=(6,0))

        def sep():
            tk.Frame(p, height=1, bg="gray").pack(fill=tk.X, pady=5)

        tk.Label(p, text="Pathfinding Visualizer", font=("Arial",13,"bold"),
                 bg="#f0f0f0").pack(pady=(0,8))

        # Grid size
        label("Grid Size")
        sf = tk.Frame(p, bg="#f0f0f0"); sf.pack(anchor="w")
        tk.Label(sf, text="Rows:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.rows_v = tk.IntVar(value=self.rows)
        tk.Spinbox(sf, from_=5, to=30, textvariable=self.rows_v, width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(sf, text="Cols:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.cols_v = tk.IntVar(value=self.cols)
        tk.Spinbox(sf, from_=5, to=40, textvariable=self.cols_v, width=4).pack(side=tk.LEFT, padx=2)
        tk.Button(p, text="Apply Size", command=self.apply_size, width=18).pack(pady=3)

        sep()

        # Random map
        label("Obstacle Density")
        self.density_v = tk.DoubleVar(value=0.3)
        tk.Scale(p, from_=0, to=0.7, resolution=0.05, variable=self.density_v,
                 orient=tk.HORIZONTAL, length=160, bg="#f0f0f0").pack(anchor="w")
        tk.Button(p, text="Generate Random Map", command=self.gen_map, width=18).pack(pady=3)

        sep()

        # Draw mode
        label("Draw Mode")
        self.mode_v = tk.StringVar(value="wall")
        for val, txt in [("wall","Draw Walls"),("start","Move Start"),("goal","Move Goal")]:
            tk.Radiobutton(p, text=txt, variable=self.mode_v, value=val, bg="#f0f0f0").pack(anchor="w")

        sep()

        # Algorithm & Heuristic
        label("Algorithm")
        self.algo_v = tk.StringVar(value="A*")
        for val, txt in [("A*","A* Search"),("GBFS","Greedy Best First")]:
            tk.Radiobutton(p, text=txt, variable=self.algo_v, value=val, bg="#f0f0f0").pack(anchor="w")

        label("Heuristic")
        self.heur_v = tk.StringVar(value="Manhattan")
        for val in ["Manhattan","Euclidean"]:
            tk.Radiobutton(p, text=val, variable=self.heur_v, value=val, bg="#f0f0f0").pack(anchor="w")

        sep()

        # Dynamic mode
        self.dynamic_v = tk.BooleanVar(value=False)
        tk.Checkbutton(p, text="Dynamic Mode (random obstacles)",
                       variable=self.dynamic_v, bg="#f0f0f0").pack(anchor="w")

        sep()

        # Action buttons
        tk.Button(p, text="Start Search", command=self.start_search,
                  bg="green", fg="white", width=18).pack(pady=2)
        tk.Button(p, text="Stop", command=self.stop,
                  bg="red", fg="white", width=18).pack(pady=2)
        tk.Button(p, text="Reset", command=self.reset, width=18).pack(pady=2)

        sep()

        # Metrics
        label("Metrics")
        self.lbl_nodes = tk.Label(p, text="Nodes Visited: 0", bg="#f0f0f0", anchor="w")
        self.lbl_cost  = tk.Label(p, text="Path Cost: 0",     bg="#f0f0f0", anchor="w")
        self.lbl_time  = tk.Label(p, text="Time: 0 ms",       bg="#f0f0f0", anchor="w")
        for lbl in [self.lbl_nodes, self.lbl_cost, self.lbl_time]:
            lbl.pack(anchor="w")

        sep()

        # Legend
        label("Legend")
        for color, txt in [("green","Start"),("red","Goal"),("black","Wall"),
                            ("yellow","Frontier"),("lightblue","Visited"),
                            ("orange","Path"),("purple","Agent")]:
            row = tk.Frame(p, bg="#f0f0f0"); row.pack(anchor="w", pady=1)
            tk.Frame(row, bg=color, width=14, height=14).pack(side=tk.LEFT, padx=(0,5))
            tk.Label(row, text=txt, bg="#f0f0f0", font=("Arial",9)).pack(side=tk.LEFT)

    # ---------- Canvas ----------
    def build_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.cols*CELL,
                                height=self.rows*CELL, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    # ---------- Draw ----------
    def draw(self, visited=None, path=None, agent=None):
        visited  = visited or set()
        path_set = set(path or [])
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c*CELL, r*CELL
                x2, y2 = x1+CELL, y1+CELL
                ct = self.grid[r][c]
                pos = (r, c)
                if ct == WALL:         color = COLORS[WALL]
                elif ct == START:      color = COLORS[START]
                elif ct == GOAL:       color = COLORS[GOAL]
                elif pos == agent:     color = COLORS["agent"]
                elif pos in path_set:  color = COLORS["path"]
                elif pos in visited:   color = COLORS["visited"]
                else:                  color = COLORS[EMPTY]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                if ct == START:
                    self.canvas.create_text(x1+CELL//2, y1+CELL//2, text="S", font=("Arial",9,"bold"))
                elif ct == GOAL:
                    self.canvas.create_text(x1+CELL//2, y1+CELL//2, text="G", font=("Arial",9,"bold"), fill="white")

    # ---------- Mouse ----------
    def get_cell(self, e):
        r, c = e.y//CELL, e.x//CELL
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def on_click(self, e):
        if self.running: return
        pos = self.get_cell(e)
        if not pos: return
        r, c = pos
        mode = self.mode_v.get()
        if mode == "wall":
            self.grid[r][c] = EMPTY if self.grid[r][c] == WALL else WALL
        elif mode == "start":
            self.grid[self.start[0]][self.start[1]] = EMPTY
            self.start = pos; self.grid[r][c] = START
        elif mode == "goal":
            self.grid[self.goal[0]][self.goal[1]] = EMPTY
            self.goal = pos; self.grid[r][c] = GOAL
        self.draw()

    def on_drag(self, e):
        if self.running: return
        pos = self.get_cell(e)
        if pos and self.mode_v.get() == "wall":
            r, c = pos
            if self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
                self.draw()

    # ---------- Buttons ----------
    def apply_size(self):
        self.stop()
        self.rows, self.cols = self.rows_v.get(), self.cols_v.get()
        self.start, self.goal = (0,0), (self.rows-1, self.cols-1)
        self.grid = self.empty_grid()
        self.canvas.config(width=self.cols*CELL, height=self.rows*CELL)
        self.draw()

    def gen_map(self):
        self.stop()
        self.grid = self.empty_grid()
        density = self.density_v.get()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in (self.start, self.goal) and random.random() < density:
                    self.grid[r][c] = WALL
        self.draw()

    def reset(self):
        self.stop()
        self.grid = self.empty_grid()
        self.lbl_nodes.config(text="Nodes Visited: 0")
        self.lbl_cost.config(text="Path Cost: 0")
        self.lbl_time.config(text="Time: 0 ms")
        self.draw()

    def stop(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    # ---------- Search ----------
    def start_search(self):
        self.stop()
        t = time.time()
        path, visited = search(self.grid, self.rows, self.cols,
                               self.start, self.goal,
                               self.algo_v.get(), self.heur_v.get())
        elapsed = (time.time()-t)*1000

        self.lbl_nodes.config(text=f"Nodes Visited: {len(visited)}")
        self.lbl_time.config(text=f"Time: {elapsed:.2f} ms")

        if path is None:
            self.draw(visited=visited)
            self.lbl_cost.config(text="Path Cost: N/A")
            messagebox.showwarning("No Path", "No path found!")
            return

        self.lbl_cost.config(text=f"Path Cost: {len(path)-1}")
        self.draw(visited=visited, path=path)
        self.path, self.path_idx, self.visited = path, 0, visited
        self.running = True
        self.animate()

    # ---------- Animation ----------
    def animate(self):
        if not self.running: return
        if self.path_idx >= len(self.path):
            self.running = False
            return
        self.agent = self.path[self.path_idx]
        self.path_idx += 1
        self.draw(visited=self.visited, path=self.path, agent=self.agent)
        if self.dynamic_v.get():
            self.spawn_obstacle()
        self.after_id = self.root.after(150, self.animate)

    # ---------- Dynamic Obstacles ----------
    def spawn_obstacle(self):
        if random.random() > 0.10: return
        r, c = random.randint(0,self.rows-1), random.randint(0,self.cols-1)
        pos = (r, c)
        if self.grid[r][c] != EMPTY or pos in (self.start, self.goal, self.agent):
            return
        self.grid[r][c] = WALL
        if pos in self.path[self.path_idx:]:
            self.replan()

    def replan(self):
        path, new_visited = search(self.grid, self.rows, self.cols,
                                   self.agent, self.goal,
                                   self.algo_v.get(), self.heur_v.get())
        if path is None:
            self.running = False
            messagebox.showwarning("Blocked!", "No path left after obstacle!")
            return
        self.visited.update(new_visited)
        self.path, self.path_idx = path, 0
        self.lbl_nodes.config(text=f"Nodes Visited: {len(self.visited)}")
        self.lbl_cost.config(text=f"Path Cost: {len(self.path)-1}")


# ---------- Run ----------
root = tk.Tk()
App(root)
root.mainloop()