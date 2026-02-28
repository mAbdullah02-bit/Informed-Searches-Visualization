import tkinter as tk
from tkinter import messagebox
import heapq
import random
import math
import time

# grid cell size in pixels
CELL_SIZE = 30

# grid cell types
EMPTY = 0
WALL  = 1
START = 2
GOAL  = 3

# colors for each cell type and state
COLOR = {
    "empty":    "white",
    "wall":     "black",
    "start":    "green",
    "goal":     "red",
    "visited":  "lightblue",
    "frontier": "yellow",
    "path":     "orange",
    "agent":    "purple"
}

# -------------------------------------------------------
# Heuristic functions
# -------------------------------------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_heuristic(name):
    if name == "Manhattan":
        return manhattan
    return euclidean

# -------------------------------------------------------
# Get walkable neighbors of a cell
# -------------------------------------------------------
def get_neighbors(grid, rows, cols, cell):
    r, c = cell
    neighbors = []
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            neighbors.append((nr, nc))
    return neighbors

# -------------------------------------------------------
# A* Search
# -------------------------------------------------------
def astar(grid, rows, cols, start, goal, heuristic_name):
    h = get_heuristic(heuristic_name)
    open_list = []
    heapq.heappush(open_list, (h(start, goal), 0, start))
    came_from = {}
    g = {start: 0}
    visited = set()
    frontier = set([start])

    while open_list:
        f_val, g_val, current = heapq.heappop(open_list)
        frontier.discard(current)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, visited

        for nb in get_neighbors(grid, rows, cols, current):
            new_g = g[current] + 1
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                f = new_g + h(nb, goal)
                heapq.heappush(open_list, (f, new_g, nb))
                came_from[nb] = current
                frontier.add(nb)

    return None, visited

# -------------------------------------------------------
# Greedy Best First Search
# -------------------------------------------------------
def gbfs(grid, rows, cols, start, goal, heuristic_name):
    h = get_heuristic(heuristic_name)
    open_list = []
    heapq.heappush(open_list, (h(start, goal), start))
    came_from = {}
    visited = set()
    seen = set([start])
    frontier = set([start])

    while open_list:
        _, current = heapq.heappop(open_list)
        frontier.discard(current)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, visited

        for nb in get_neighbors(grid, rows, cols, current):
            if nb not in seen:
                seen.add(nb)
                heapq.heappush(open_list, (h(nb, goal), nb))
                came_from[nb] = current
                frontier.add(nb)

    return None, visited
