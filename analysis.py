import numpy as np
import matplotlib.pyplot as plt
import os

def print_distribution(dataset,
    channel_names=None):
    N, C, H, W = dataset.shape

    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(C)]
        
    for c in range(C):
        channel = dataset[:, c, :, :]
        mean = channel.mean()
        std  = channel.std()
        minv = channel.min()
        maxv = channel.max()
        print("========================================")
        print("Name:" + channel_names[c])
        print("std", f"{std:.6f}") # std ≈ 1 or range ≈ [-1, 1]
        print("mean", f"{mean:.6f}") # mean ≈ 0
        print("min", f"{minv:.6f}")
        print("max", f"{maxv:.6f}")
        print()

# data: [Y,X]
def plot_data(data1, data2=None, data1_title="", data2_title=""):
    data1 = np.rot90(data1, k=2) # rotate 180 degree
    data1 = np.fliplr(data1) # flip horizontal
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    limit1 = max(abs(data1.min()), abs(data1.max()))
    # axes[0].imshow(data1, cmap='RdBu_r', vmin=-limit1, vmax=limit1)
    axes[0].imshow(data1, cmap='RdBu_r')
    axes[0].set_title(data1_title)
    axes[0].axis('off')
    if data2 is not None:
        data2 = np.rot90(data2, k=2) # rotate 180 degree
        data2 = np.fliplr(data2) # flip horizontal
        limit2 = max(abs(data1.min()), abs(data1.max()))
        # axes[1].imshow(data2, cmap='RdBu_r', vmin=-limit2, vmax=limit2)
        axes[1].imshow(data2, cmap='RdBu_r')
        axes[1].set_title(data2_title)
        axes[1].axis('off')
    
    plt.show()
    # plt.close()

def plot_data_grid(datasets, titles=None, cols=3):
    n = len(datasets)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Flatten axes for easy iteration
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < n:
            # Apply your transformations
            data = np.fliplr(np.rot90(datasets[i], k=2))
            axes[i].imshow(data, cmap='RdBu_r')
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # plt.tight_layout()
    plt.show()

def plot_distributions(
    dataset,
    channel_names=None,
    bins=200,
    clip_percentile=99.5
):
    """
    dataset: np.ndarray of shape [N, C, H, W]
    """
    N, C, H, W = dataset.shape

    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(C)]

    fig, axes = plt.subplots(2, (C + 1) // 2, figsize=(4 * C, 6))
    axes = axes.flatten()

    for c in range(C):
        channel = dataset[:, c, :, :]
        mean = channel.mean()
        std  = channel.std()
        minv = channel.min()
        maxv = channel.max()
        
        data = channel.reshape(-1)

        # Optional clipping to remove extreme outliers
        lo, hi = np.percentile(data, [100 - clip_percentile, clip_percentile])
        data = np.clip(data, lo, hi)

        # max_abs_val = max(data.max(), -data.min())
        # axes[c].set_xlim(-max_abs_val, max_abs_val)
        axes[c].hist(data, bins=bins, density=True)
        axes[c].set_title(channel_names[c])
        axes[c].set_xlabel("Value")
        axes[c].set_ylabel("Density")

    # Remove unused axes
    for i in range(C, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
