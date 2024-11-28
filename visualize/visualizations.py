# load packages
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



def plot_optimization_path(data, theta_history, theta_true, loss_func, title="Optimization Landscape", func_args=None):
    """
    Plot the optimization path and loss landscape with dynamic domain calculation.
    
    Parameters:
    - data: Training data points
    - theta_history: History of theta values during optimization
    - theta_true: True parameter value
    - loss_func: Loss function to visualize
    - title: Plot title
    - func_args: Optional dictionary of additional arguments for the loss function
    """
    # Convert theta_history to numpy array
    theta_path = torch.stack(theta_history).numpy()
    
    # Dynamically determine the plot range
    # Include both theta_path and theta_true in the range calculation
    all_points = np.vstack([theta_path, theta_true.numpy()])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # Add padding (20% of the range)
    range_vals = max_vals - min_vals
    padding = range_vals * 0.2
    plot_min = min_vals - padding
    plot_max = max_vals + padding
    
    # Create a grid of theta values using the dynamic range
    n_points = 50
    theta1 = np.linspace(plot_min[0], plot_max[0], n_points)
    theta2 = np.linspace(plot_min[1], plot_max[1], n_points)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)
    
    # Sample a subset of data points for faster computation
    num_samples = min(50, len(data))
    sampled_indices = np.random.choice(len(data), num_samples, replace=False)
    sampled_data = [data[i] for i in sampled_indices]
    
    # Calculate loss for each point in the grid
    Z = np.zeros_like(Theta1)
    for i in range(len(theta1)):
        for j in range(len(theta2)):
            theta_ij = torch.tensor([Theta1[i,j], Theta2[i,j]], dtype=torch.float32)
            # Compute average loss over sampled data points
            losses = []
            for x_i, y_i in sampled_data:
                if func_args is not None:
                    loss = loss_func(theta_ij, x_i, y_i, **func_args)
                else:
                    loss = loss_func(theta_ij, x_i, y_i)
                losses.append(loss.item())
            Z[i,j] = np.mean(losses)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot contour with more levels for better resolution
    contour = plt.contour(Theta1, Theta2, Z, levels=30, alpha=0.6)
    
    # Plot optimization path with color gradient
    points = theta_path.reshape(-1, 2)
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
    
    # Create a color gradient for the path
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap='cool', norm=norm)
    lc.set_array(np.linspace(0, len(segments), len(segments)))
    plt.gca().add_collection(lc)
    
    # Plot start and end points
    plt.plot(theta_path[0, 0], theta_path[0, 1], 'r*', markersize=15, label='Start')
    plt.plot(theta_path[-1, 0], theta_path[-1, 1], 'g*', markersize=15, label='End')
    plt.plot(theta_true[0], theta_true[1], 'y*', markersize=15, label='True θ')
    
    # Set axis labels and title
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.colorbar(lc, label='Iteration')
    
    return plt

def visualize_optimization(data, theta_history, theta_true, loss_func, scenario_name, func_args=None):
    """
    Wrapper function to create both loss landscape and loss history visualizations
    
    Parameters:
    - data: Training data points
    - theta_history: History of theta values during optimization
    - theta_true: True parameter value
    - loss_func: Loss function to visualize
    - scenario_name: Name of the optimization scenario
    - func_args: Optional dictionary of additional arguments for the loss function
    """
    # Plot the optimization path
    plt = plot_optimization_path(
        data,
        theta_history,
        theta_true,
        loss_func,
        f"{scenario_name} Optimization Landscape",
        func_args
    )
    
    # Plot the distance to true theta over iterations in a separate figure
    plt.figure(figsize=(8, 4))
    distance_values = [torch.norm(theta - theta_true).item() for theta in theta_history]
    plt.plot(distance_values)
    plt.xlabel('Iteration')
    plt.ylabel(r'$||\theta_{\text{true}} - \theta_t||$')
    plt.title(f'{scenario_name} Distance to True Theta Over Time')
    plt.grid(True)
    plt.show()

def plot_theta_dist(theta_history, theta_true, scenario_name):
    """
    Plot the distance between current theta and true theta over iterations
    """
    plt.figure(figsize=(8,4))
    distance_values = [torch.norm(theta-theta_true).item() for theta in theta_history]
    plt.plot(distance_values)
    plt.xlabel('Iteration')
    plt.ylabel(r'$||\theta_{\text{true}} - \theta_t||$')
    plt.title(f'{scenario_name} Distance to True Theta Over Time')
    plt.grid(True)
    plt.show()


def compare_convergence(theta_true_1, theta_true_2, conv_res_1, conv_res_2, title_1, title_2, opt_type=""):
# Plot both results
    plt.figure(figsize=(7, 7))
    
    # Distance to optimal plot
    dist1 = [torch.norm(theta - theta_true_1).item()
                  for theta in conv_res_1]
    dist2 = [torch.norm(theta - theta_true_2).item()
                  for theta in conv_res_2]
    
    plt.plot(dist1, label=title_1, color='blue')
    plt.plot(dist2, label=title_2, color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Optimal θ')
    plt.title(opt_type + ' Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale to better see convergence rates
    plt.show()