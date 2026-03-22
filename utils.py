"""
Utility functions for Diffusion Models Tutorial

This module provides helper functions for:
- Loading configuration
- Normalizing images for visualization
- Creating animations of the denoising process
"""

import torch
import numpy as np
import yaml
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os


def load_config(config_file):
    """
    Load configuration from YAML file

    Args:
        config_file: path to YAML config file
    Returns:
        dict: configuration parameters
    """
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def unorm(x):
    """
    Normalize image to [0, 1] range

    Args:
        x: numpy array of shape (h, w, 3)
    Returns:
        normalized array in [0, 1]
    """
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)


def norm_all(store, n_t, n_s):
    """
    Apply unity normalization to all timesteps of all samples

    Args:
        store: array of images
        n_t: number of timesteps
        n_s: number of samples
    Returns:
        normalized array
    """
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all):
    """
    Normalize torch tensors for visualization

    Args:
        x_all: tensor of shape (n_samples, 3, h, w)
    Returns:
        normalized tensor in [0, 1] range
    """
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)


def plot_grid(x, n_sample, n_rows, save_dir, filename):
    """
    Plot a grid of images and save to file

    Args:
        x: tensor of images (n_sample, 3, h, w)
        n_sample: number of samples
        n_rows: number of rows in grid
        save_dir: directory to save image
        filename: name for saved file
    Returns:
        grid image
    """
    ncols = n_sample // n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")
    save_image(grid, save_path)
    print(f'Saved image at {save_path}')
    return grid


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False):
    """
    Create an animation showing the denoising process over time

    This creates a grid of images that shows how samples evolve
    from noise to final images across multiple timesteps.

    Args:
        x_gen_store: array of shape (n_timesteps, n_sample, 3, h, w)
        n_sample: number of samples to display
        nrows: number of rows in the grid
        save_dir: directory to save animation (if save=True)
        fn: filename for saving
        w: additional parameter for filename
        save: whether to save the animation to disk

    Returns:
        matplotlib animation object
    """
    ncols = n_sample // nrows

    # Convert from (n_t, n_s, C, H, W) to (n_t, n_s, H, W, C) for display
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)

    # Normalize all frames to [0, 1] for visualization
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)

    # Create figure and subplots
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(ncols*4, nrows*4)
    )

    # Make sure axs is 2D even for single row/col
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    def animate_diff(i, store):
        """Update function for animation"""
        print(f'Animating frame {i+1}/{store.shape[0]}', end='\\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                img = store[i, (row * ncols) + col]
                plots.append(axs[row, col].imshow(img))
        return plots

    # Create animation
    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[nsx_gen_store],
        interval=200,  # milliseconds between frames
        blit=False,
        repeat=True,
        frames=nsx_gen_store.shape[0]
    )

    plt.close()

    # Optionally save to file
    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{fn}_w{w}.gif")
        ani.save(save_path, dpi=100, writer=PillowWriter(fps=5))
        print(f'\\nSaved animation at {save_path}')

    return ani