import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_points(filename):
    """Load points from a file."""
    return np.loadtxt(filename)

def plot_points(ax, points, title, color='b', alpha=0.5):
    """Plot points in 3D."""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=color, alpha=alpha, s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    # Load all sets of points
    original = load_points('original_points.txt')
    inscribed = load_points('inscribed_ellipsoid_points.txt')
    min_sampling = load_points('min_sampling_points.txt')
    svd = load_points('svd_points.txt')

    # Create a 2x2 subplot for 3D visualization
    fig = plt.figure(figsize=(15, 15))
    
    # Original points
    ax1 = fig.add_subplot(221, projection='3d')
    plot_points(ax1, original, 'Original Points', 'b')
    
    # Inscribed ellipsoid points
    ax2 = fig.add_subplot(222, projection='3d')
    plot_points(ax2, inscribed, 'Inscribed Ellipsoid', 'r')
    
    # Min sampling points
    ax3 = fig.add_subplot(223, projection='3d')
    plot_points(ax3, min_sampling, 'Min Sampling', 'g')
    
    # SVD points
    ax4 = fig.add_subplot(224, projection='3d')
    plot_points(ax4, svd, 'SVD Rounding', 'm')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('rounding_comparison_grid.png')
    plt.close()

    # Create a combined 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points with different colors
    ax.scatter(original[:, 0], original[:, 1], original[:, 2], 
              c='b', alpha=0.3, s=1, label='Original')
    ax.scatter(inscribed[:, 0], inscribed[:, 1], inscribed[:, 2], 
              c='r', alpha=0.3, s=1, label='Inscribed')
    ax.scatter(min_sampling[:, 0], min_sampling[:, 1], min_sampling[:, 2], 
              c='g', alpha=0.3, s=1, label='Min Sampling')
    ax.scatter(svd[:, 0], svd[:, 1], svd[:, 2], 
              c='m', alpha=0.3, s=1, label='SVD')
    
    ax.set_title('All Rounding Methods Comparison')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.savefig('rounding_comparison_combined.png')
    plt.close()
    
    print("Visualization files have been saved as 'rounding_comparison_grid.png' and 'rounding_comparison_combined.png'")

if __name__ == "__main__":
    main() 