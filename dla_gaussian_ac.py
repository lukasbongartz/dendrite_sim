import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

size = (100, 200)
n_particles = int(size[0] * size[1] * 0.2)
n_iter = 10000
p_stick = 0.8
alpha_max = 0
field_flip_interval = 1000
rng = np.random.default_rng()

field = np.zeros(size, dtype=np.int8)
particles_pos = np.unravel_index(
    np.random.choice(np.arange(size[0] * size[1]), n_particles, replace=False),
    size
)
field[particles_pos] = 1

stucked_field = np.zeros(size, dtype=np.int8)
mid_y = size[0] // 2
stucked_field[mid_y, 1] = 1
stucked_field[mid_y, size[1] - 2] = 1

field_b_even = field.reshape((-1, 2, field.shape[1] // 2, 2))
field_blocks_even = field_b_even.transpose((0, 2, 1, 3))
field_b_odd = field[1:-1, 1:-1].reshape((-1, 2, field.shape[1] // 2 - 1, 2))
field_blocks_odd = field_b_odd.transpose((0, 2, 1, 3))

frames = []
occupied_frames = []

def gaussian_decay(y, y_mid, sigma):
    return np.exp(-((y - y_mid) ** 2) / (2 * sigma ** 2))

def apply_gaussian_ac_bias(field, E_field, alpha_max):
    N_y, N_x = field.shape
    mid_y = N_y // 2
    sigma = N_y / 10
    
    for y in range(N_y):
        alpha_y = alpha_max * gaussian_decay(y, mid_y, sigma)
        p_left = 0.5 * (1 - alpha_y * E_field[0])
        p_right = 0.5 * (1 + alpha_y * E_field[0])
        bias_direction = rng.choice(['left', 'right'], p=[p_left, p_right])
        if bias_direction == 'left':
            field[y] = np.roll(field[y], shift=-1)
        else:
            field[y] = np.roll(field[y], shift=1)
    return field

def probabilistic_stick(field, stucked_field, p_stick, rng):
    free_core = (field[1:-1, 1:-1] == 1)
    neighbors_down = stucked_field[2:, 1:-1] == 1
    neighbors_up = stucked_field[:-2, 1:-1] == 1
    neighbors_left = stucked_field[1:-1, :-2] == 1
    neighbors_right = stucked_field[1:-1, 2:] == 1

    can_stick = free_core & (neighbors_down | neighbors_up | neighbors_left | neighbors_right)
    stick_random = (rng.random(can_stick.shape) < p_stick)
    final_stick = can_stick & stick_random

    field[1:-1, 1:-1][final_stick] = 0
    stucked_field[1:-1, 1:-1][final_stick] = 1

def construct_graph(stucked_field):
    G = nx.Graph()
    rows, cols = np.where(stucked_field == 1)
    for i, j in zip(rows, cols):
        G.add_node((i, j))
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < stucked_field.shape[0] and 0 <= nj < stucked_field.shape[1]:
                if stucked_field[ni, nj] == 1:
                    G.add_edge((i, j), (ni, nj))
    return G

def compute_degree_grid(stucked_field):
    G = construct_graph(stucked_field)
    degree_map = nx.degree_centrality(G)
    grid = np.zeros(stucked_field.shape, dtype=float)
    for (r, c), val in degree_map.items():
        grid[r, c] = val
    return grid

for i in tqdm(range(n_iter), desc="Simulating Gaussian AC Field + DLA"):
    E_field = np.array([1, 0]) if ((i // field_flip_interval) % 2 == 0) else np.array([-1, 0])

    if i % 2 == 0:
        rng.shuffle(field_blocks_even, axis=3)
        rng.shuffle(field_blocks_even, axis=2)
    else:
        rng.shuffle(field_blocks_odd, axis=3)
        rng.shuffle(field_blocks_odd, axis=2)

    field = apply_gaussian_ac_bias(field, E_field, alpha_max)
    probabilistic_stick(field, stucked_field, p_stick, rng)

    if i % 50 == 0:
        frames.append(i)
        degree_grid = compute_degree_grid(stucked_field)
        min_val = degree_grid.min()
        max_val = degree_grid.max()
        
        if max_val > min_val:
            deg_norm = (degree_grid - min_val) / (max_val - min_val)
        else:
            deg_norm = np.zeros_like(degree_grid)

        combined_view = np.where(
            stucked_field == 1,
            deg_norm,
            np.where(field == 1, 0.6, 0.0)
        )
        occupied_frames.append(combined_view)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(occupied_frames[0], cmap='viridis', interpolation='nearest',
               origin='lower', vmin=0, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.03)
plt.colorbar(im, cax=cax, label='$C_D(i)$ (norm.)')

ax.set_xlim(0, size[1])
ax.set_ylim(0, size[0])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

def update(frame_num):
    frame_idx = frames.index(frame_num)
    im.set_data(occupied_frames[frame_idx])
    return [im]

samples_dir = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(samples_dir, exist_ok=True)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
ani.save(os.path.join(samples_dir, "dla_gaussian_ac.gif"), writer='pillow', fps=20, dpi=300)
plt.close()