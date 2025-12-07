import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

import matplotlib.pyplot as plt

def visualize_depth(filepath):
    """Load and visualize depth data from a .npz file."""
    # Load the npz file
    data = np.load(filepath)
    
    # Get the depth array (adjust key name if different)
    depth = data['depth'] if 'depth' in data.files else data[data.files[0]]
    # Ensure float array
    depth = depth.astype(np.float32)
    
    # Build validity mask (treat 0 or negative as invalid, also NaNs)
    invalid = np.isnan(depth) | (depth <= 0)
    valid = ~invalid
    
    if not np.any(valid):
        print("No valid depth values found (all zeros/NaNs/negatives).")
        return
    
    # Optionally normalize valid values to [0,1] if outside
    vmin_raw = float(np.nanmin(depth[valid]))
    vmax_raw = float(np.nanmax(depth[valid]))
    
    # Contrast stretch using percentiles to avoid outliers
    p2 = float(np.nanpercentile(depth[valid], 2))
    p98 = float(np.nanpercentile(depth[valid], 98))
    low = max(vmin_raw, p2)
    high = min(vmax_raw, p98) if p98 > low else vmax_raw
    
    # Create a masked array for nicer visualization of invalid pixels
    depth_masked = np.ma.masked_array(depth, mask=invalid)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D heatmap (mask invalid, apply contrast limits)
    im = axes[0].imshow(depth_masked, cmap='viridis', vmin=low, vmax=high)
    axes[0].set_title('Depth Map (2D)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Depth')
    
    # Make invalid pixels transparent
    alpha = (~invalid).astype(np.float32)
    im.set_alpha(alpha)
    
    # 3D surface plot
    ax3d = fig.add_subplot(122, projection='3d')
    h, w = depth.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Downsample for performance on large arrays
    ds = max(1, int(max(h, w) // 400))
    X_ds = X[::ds, ::ds]
    Y_ds = Y[::ds, ::ds]
    depth_ds = depth.copy()[::ds, ::ds]
    depth_ds[invalid[::ds, ::ds]] = np.nan
    
    ax3d.plot_surface(X_ds, Y_ds, depth_ds, cmap='viridis')
    ax3d.set_title('Depth Map (3D)')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Depth')
    
    plt.tight_layout()
    # save
    plt.savefig('depth_visualization.png', dpi=200)

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'depth.npz'
    visualize_depth(filepath)