import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from matplotlib.colors import ListedColormap
from collections import deque

# ====== CONSTANTS ======
GRID_SIZE = 40
SENSOR_RADIUS = 6
MAX_STEPS = 20
ALPHA = 1.0  # weight for cost in scoring

# ====== CELL STATES ======
UNEXPLORED, EXPLORED, FRONTIER, OBSTACLE, ROBOT1, ROBOT2, PATH1, PATH2 = range(8)

# ====== COLOR MAPPING ======
COLORS = {
    UNEXPLORED: "#f0f0f0",   # light gray
    EXPLORED:   "#b2dfdb",   # teal-ish
    FRONTIER:   "#ffeb3b",   # yellow
    OBSTACLE:   "#211121",   # black
    ROBOT1:     "#e53935",   # red
    ROBOT2:     "#1e88e5",   # blue
    PATH1:      "#ef9a9a",   # faded red
    PATH2:      "#90caf9",   # faded blue
}
COLOR_LIST = [COLORS[i] for i in range(len(COLORS))]

# ====== ROBOT CLASS ======
class Robot:
    def __init__(self, rid, pos):
        self.rid = rid
        self.position = pos
        self.path = [pos]
        self.target = None

    def sense(self, grid):
        x, y = self.position
        for i in range(max(0, x - SENSOR_RADIUS), min(GRID_SIZE, x + SENSOR_RADIUS + 1)):
            for j in range(max(0, y - SENSOR_RADIUS), min(GRID_SIZE, y + SENSOR_RADIUS + 1)):
                if grid[i, j] != OBSTACLE:
                    grid[i, j] = EXPLORED

    def detect_frontiers(self, grid):
        frontiers = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i, j] == EXPLORED:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+dx, j+dy
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] == UNEXPLORED:
                            frontiers.append((ni, nj))
        return list(set(frontiers))

    def choose_target(self, frontiers, utility_grid):
        scores = []
        for cell in frontiers:
            u = utility_grid[cell]
            cost = abs(self.position[0] - cell[0]) + abs(self.position[1] - cell[1])
            scores.append((cell, u - ALPHA * cost))
        if not scores:
            return None
        max_score = max(scores, key=lambda x: x[1])[1]
        best = [c for c, s in scores if s == max_score]
        return random.choice(best)

    def step(self, grid):
        if not self.target:
            return
        x, y = self.position
        tx, ty = self.target
        moves = []
        if tx > x and grid[x+1, y] != OBSTACLE: moves.append((x+1, y))
        if tx < x and grid[x-1, y] != OBSTACLE: moves.append((x-1, y))
        if ty > y and grid[x, y+1] != OBSTACLE: moves.append((x, y+1))
        if ty < y and grid[x, y-1] != OBSTACLE: moves.append((x, y-1))
        if moves:
            self.position = min(moves, key=lambda c: abs(c[0]-tx) + abs(c[1]-ty))
            self.path.append(self.position)
            if self.position == self.target:
                self.target = None

# ====== ENVIRONMENT ======
def add_obstacles(grid):
    grid[15:20, 0:20] = OBSTACLE
    grid[30:35, 20:35] = OBSTACLE
    grid[5:10, 25:27] = OBSTACLE
    grid[8:10, 25:30] = OBSTACLE
    grid[30:35, 10:12] = OBSTACLE
    grid[33:35, 10:16] = OBSTACLE
    grid[5:10, 5:10] = OBSTACLE
    grid[17:22, 30:35] = OBSTACLE

# ====== COST MAP FUNCTION ======
def compute_cost_map(grid, start):
    cost_map = np.full(grid.shape, np.inf)
    from collections import deque
    dq = deque([start])
    cost_map[start] = 0
    while dq:
        x, y = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] != OBSTACLE:
                if cost_map[nx, ny] > cost_map[x, y] + 1:
                    cost_map[nx, ny] = cost_map[x, y] + 1
                    dq.append((nx, ny))
    return cost_map

# ====== SIMULATION ======
def simulate():
    grid = np.full((GRID_SIZE, GRID_SIZE), UNEXPLORED)
    add_obstacles(grid)
    r1 = Robot(1, (GRID_SIZE//2, GRID_SIZE//2))
    r2 = Robot(2, (GRID_SIZE//2, GRID_SIZE//2))

    r1.sense(grid)
    r2.sense(grid)

    frames = []
    for _ in range(MAX_STEPS):
        frontiers = r1.detect_frontiers(grid) + r2.detect_frontiers(grid)
        frontiers = list(set(frontiers))

        utility = np.zeros_like(grid, dtype=float)
        for f in frontiers:
            utility[f] = 1.0

        if not r1.target:
            r1.target = r1.choose_target(frontiers, utility)
            if r1.target:
                for f in frontiers:
                    d = abs(f[0] - r1.target[0]) + abs(f[1] - r1.target[1])
                    if d <= SENSOR_RADIUS:
                        utility[f] *= max(1 - d/SENSOR_RADIUS, 0.0)
        if not r2.target:
            r2.target = r2.choose_target(frontiers, utility)
            if r2.target:
                for f in frontiers:
                    d = abs(f[0] - r2.target[0]) + abs(f[1] - r2.target[1])
                    if d <= SENSOR_RADIUS:
                        utility[f] *= max(1 - d/SENSOR_RADIUS, 0.0)

        r1.step(grid)
        r2.step(grid)

        r1.sense(grid)
        r2.sense(grid)

        # Visualization
        vis = grid.copy()
        for x, y in r1.path: vis[x, y] = PATH1
        for x, y in r2.path: vis[x, y] = PATH2
        vis[r1.position] = ROBOT1
        vis[r2.position] = ROBOT2
        # Overlay frontiers regardless of explored state
        for fx, fy in frontiers:
            if vis[fx, fy] not in (OBSTACLE, ROBOT1, ROBOT2, PATH1, PATH2):
                vis[fx, fy] = FRONTIER

        frames.append(vis)

    cost_r1 = compute_cost_map(grid, r1.position)
    cost_r2 = compute_cost_map(grid, r2.position)

    return frames, grid, r1, r2, cost_r1, cost_r2

# ====== ANIMATION ======
def animate(frames):
    fig, ax = plt.subplots(figsize=(7,7))
    cmap = ListedColormap(COLOR_LIST)
    img = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=len(COLOR_LIST)-1)
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        img.set_data(frames[i])
        return (img,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False, repeat=False)
    plt.show()
    return anim

# ====== PLOT COST MAPS ======
def plot_cost_maps(grid, cost1, cost2):
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    im1 = axs[0].imshow(cost1, origin='upper')
    axs[0].set_title('Cost Map from Robot1 Final Pos')
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(cost2, origin='upper')
    axs[1].set_title('Cost Map from Robot2 Final Pos')
    fig.colorbar(im2, ax=axs[1])

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

if __name__ == '__main__':
    frames, final_grid, r1, r2, cost_r1, cost_r2 = simulate()
    anim = animate(frames)
    plot_cost_maps(final_grid, cost_r1, cost_r2)
