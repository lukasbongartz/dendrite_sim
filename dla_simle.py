import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def run_dla_simulation(size=(100, 100), n_particles=1000, n_iter=1000):
    """Diffusion-limited aggregation using Margolus shuffling"""
    rng = np.random.default_rng()
    
    # Initialize
    field = np.zeros(size, dtype=np.int8)
    particles_pos = np.unravel_index(
        np.random.choice(np.arange(size[0] * size[1]), n_particles, replace=False), size
    )
    field[particles_pos] = 1

    sticked_field = np.zeros(size, dtype=np.int8)
    sticked_field[size[0]//2, size[1]//2] = 1  # Seed

    # Margolus 
    field_b_even = field.reshape((-1, 2, field.shape[1]//2, 2))
    field_blocks_even = field_b_even.transpose((0, 2, 1, 3))
    field_b_odd = field[1:-1, 1:-1].reshape((-1, 2, field.shape[1]//2-1, 2))
    field_blocks_odd = field_b_odd.transpose((0, 2, 1, 3))

    frames = []
    occupied_frames = []


    for i in range(n_iter):
        # Alternate even vs odd blocks
        if i % 2 == 0:
            rng.shuffle(field_blocks_even, axis=3)
            rng.shuffle(field_blocks_even, axis=2)
        else:
            rng.shuffle(field_blocks_odd, axis=3)
            rng.shuffle(field_blocks_odd, axis=2)

        stick_1_0 = (field[2:, 1:-1] + sticked_field[1:-1, 1:-1] == 2) & (sticked_field[2:, 1:-1] == 0)
        field[2:, 1:-1][stick_1_0] = 0
        sticked_field[2:, 1:-1][stick_1_0] = 1

        stick_m1_0 = (field[0:-2, 1:-1] + sticked_field[1:-1, 1:-1] == 2) & (sticked_field[0:-2, 1:-1] == 0)
        field[0:-2, 1:-1][stick_m1_0] = 0
        sticked_field[0:-2, 1:-1][stick_m1_0] = 1

        stick_0_1 = (field[1:-1, 2:] + sticked_field[1:-1, 1:-1] == 2) & (sticked_field[1:-1, 2:] == 0)
        field[1:-1, 2:][stick_0_1] = 0
        sticked_field[1:-1, 2:][stick_0_1] = 1

        stick_0_m1 = (field[1:-1, 0:-2] + sticked_field[1:-1, 1:-1] == 2) & (sticked_field[1:-1, 0:-2] == 0)
        field[1:-1, 0:-2][stick_0_m1] = 0
        sticked_field[1:-1, 0:-2][stick_0_m1] = 1

        if i % 10 == 0:
            frames.append(i)
            combined_view = np.where(field == 1, 0.8, np.where(sticked_field == 1, 0, 1))
            occupied_frames.append(combined_view)
            
    return frames, occupied_frames

def create_animation(frames, occupied_frames, output_path):

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(occupied_frames[0], cmap='gray', interpolation='nearest', 
                   origin='lower', vmin=0, vmax=1)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('white')
    
    ax.tick_params(axis='both', which='both')
    ax.tick_params(labeltop=False, labelright=False)
    ax.tick_params(top=True, right=True)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    def update(frame_num):
        frame_idx = frames.index(frame_num)
        im.set_data(occupied_frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save(output_path, writer='pillow')
    plt.close()

if __name__ == "__main__":
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    frames, occupied_frames = run_dla_simulation()
    create_animation(frames, occupied_frames, os.path.join(samples_dir, "dla_simulation.gif"))