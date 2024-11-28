import generate_data, visualize, loss_functions, algorithms

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



seed = 302

# Generate data
data, beta_true = generate_data.generate_convex_data(N=1000, d=2, random_state=seed)

#From grid search: 

# SGD; final loss = 0.46852, parameter error = 0.03285
lr_sgd = 0.1
decay_sgd = 0.0
# mini-batch; final loss = 0.437296, parameter error = 0.384192
lr_mb = 0.1
decay_mb = 0.0
batch_size_mb = 32
# SVRG; final loss = 0.46897, parameter error = 0.0292322
lr_svrg = 0.01
n_epochs = 20
inner_loop_size = 100
batch_size_svrg = 4
# momentum; final loss = 0.467749, parameter error = 0.032880
lr_momentum = 0.001
momentum = 0.2
batch_size_momentum = 32
# ada; final loss = 0.469504, parameter error = 0.027039
lr_ada = 0.001
batch_size_ada = 32
# adam; final loss = 0.464648, parameter error = 0.044999
lr_adam = 0.001
batch_size_adam = 32
beta_1 = 0.9
beta_2 = 0.99

lr = 0.001
n_iter = 500
lambda_reg = 0.1
func_args = {}
theta_init = torch.rand_like(beta_true, requires_grad=True)
tol = 1e-06

torch.manual_seed(seed)
theta_estimated, loss_history, theta_history, time_history = algorithms.SGD(
    func=loss_functions.absolute_loss,
    lr=lr_sgd,
    n_iter=n_iter*10,
    data=data,
    theta_true=beta_true,
    tol = tol,
    batch_size=1,
    theta_init=theta_init,
    func_args=func_args,
    do_time=True
)
normed_distance = [torch.norm(theta - beta_true).item() for theta in theta_history]
print("**** SGD done! ****\n")

torch.manual_seed(seed)
theta_estimated_mb, loss_history_mb, theta_history_mb, time_history_mb = algorithms.SGD(
    func=loss_functions.absolute_loss,
    lr=lr_mb,
    n_iter=n_iter,
    data=data,
    tol = tol,
    theta_true=beta_true,
    batch_size=batch_size_mb,
    theta_init=theta_init,
    func_args=func_args,
    do_time=True
)
normed_distance_mb = [torch.norm(theta - beta_true).item() for theta in theta_history_mb]
print("**** Mini-Batch done! ****\n")

torch.manual_seed(seed)
theta_estimated_svrg, loss_history_svrg, theta_history_svrg, time_history_svrg = algorithms.SVRG(
    func=loss_functions.absolute_loss,
    lr=lr_svrg,
    n_epochs=n_epochs,
    inner_loop_size=inner_loop_size,
    data=data,
    tol = tol,
    theta_true=beta_true,
    batch_size=batch_size_svrg,
    theta_init=theta_init,
    func_args=func_args,
    do_time=True
)
normed_distance_svrg = [torch.norm(theta - beta_true).item() for theta in theta_history_svrg]
times_svrg = [time_history_svrg[(i+1)*inner_loop_size - 1] for i in range(n_epochs)]
print("**** SVRG done! ****\n")

torch.manual_seed(seed)
theta_momentum, loss_hist_momentum, theta_hist_momentum, time_history_momentum = algorithms.MomentumSGD(
    func = loss_functions.absolute_loss,
    lr = lr_momentum,
    momentum=momentum,
    n_iter=n_iter,
    data = data, 
    theta_true=beta_true,
    tol = tol,
    batch_size=batch_size_momentum,
    theta_init=theta_init,
    func_args=func_args,
    do_time=True)

normed_distance_momentum = [torch.norm(theta - beta_true).item() for theta in theta_hist_momentum]
print("**** Momentum done! **** \n")

torch.manual_seed(seed)
theta_AdaGrad, loss_hist_AdaGrad, theta_hist_AdaGrad, time_history_AdaGrad = algorithms.AdaGrad(
    func = loss_functions.absolute_loss,
    lr = lr_ada,
    n_iter=n_iter,
    data = data, 
    theta_true=beta_true,
    tol = tol,
    batch_size=batch_size_ada,
    theta_init=theta_init,
    func_args=func_args,
    do_time=True)

normed_distance_AdaGrad = [torch.norm(theta - beta_true).item() for theta in theta_hist_AdaGrad]
print("**** Adagrad done! **** \n")

torch.manual_seed(seed)
theta_Adam, loss_hist_Adam, theta_hist_Adam, time_history_Adam = algorithms.Adam(
    func = loss_functions.absolute_loss,
    lr = lr_adam,
    beta_1=beta_1,
    beta_2=beta_2,
    n_iter=n_iter,
    theta_true=beta_true,
    tol = tol, 
    data=data,
    batch_size=batch_size_adam,
    theta_init= theta_init,
    func_args=func_args,
    do_time=True)

normed_distance_Adam = [torch.norm(theta - beta_true).item() for theta in theta_hist_Adam]
print("**** Adam done! **** \n")


algorithms_data = {'SGD': {'normed_distance': normed_distance, 
                           'time_history': time_history},
                   'Mini-Batch': {'normed_distance': normed_distance_mb, 
                           'time_history': time_history_mb},
                   'SVRG': {'normed_distance': normed_distance_svrg, 
                           'time_history': times_svrg},
                   'Momentum': {'normed_distance': normed_distance_momentum, 
                           'time_history': time_history_momentum},
                   'AdaGrad': {'normed_distance': normed_distance_AdaGrad, 
                           'time_history': time_history_AdaGrad},
                   'Adam': {'normed_distance': normed_distance_Adam, 
                           'time_history': time_history_Adam}}


def validate_and_process_data(algorithm_data):
    """
    Validate the input data and print helpful information about array shapes.
    
    Parameters:
    algorithm_data (dict): Input algorithm data dictionary
    
    Returns:
    dict: Validated algorithm data
    """
    processed_data = {}
    
    for algo_name, data in algorithm_data.items():
        norm_len = len(data['normed_distance'])
        time_len = len(data['time_history'])
        
        print(f"\nAlgorithm: {algo_name}")
        print(f"Length of normed_distance: {norm_len}")
        print(f"Length of time_history: {time_len}")
        
        if norm_len != time_len:
            print(f"Warning: Length mismatch for {algo_name}!")
            # Take the minimum length to ensure arrays match
            min_len = min(norm_len, time_len)
            processed_data[algo_name] = {
                'normed_distance': np.array(data['normed_distance'][:min_len]),
                'time_history': np.array(data['time_history'][:min_len])
            }
            print(f"Truncated both arrays to length {min_len}")
        else:
            processed_data[algo_name] = {
                'normed_distance': np.array(data['normed_distance']),
                'time_history': np.array(data['time_history'])
            }
    
    return processed_data


def plot_sgd_comparison(algorithm_data, save_path=None, figsize=(12, 8)):
    """
    Plot multiple SGD algorithms' convergence over time.
    
    Parameters:
    algorithm_data (dict): Dictionary with format:
        {
            'algorithm_name': {
                'normed_distance': list of convergence values,
                'time_history': list of runtime values in seconds
            },
            ...
        }
    save_path (str, optional): Path to save the plot
    figsize (tuple): Figure size in inches
    """
    # Validate and process the data
    print("Validating input data...")
    algorithm_data = validate_and_process_data(algorithm_data)
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Get a colormap with distinct colors for up to 6 algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_data)))
    
    # Plot each algorithm
    for (algo_name, data), color in zip(algorithm_data.items(), colors):
        # Convert to numpy arrays if they aren't already
        time_history = np.array(data['time_history'])
        normed_distance = np.array(data['normed_distance'])
        
        # Sort by time to ensure proper line plotting
        sort_idx = np.argsort(time_history)
        time_history = time_history[sort_idx]
        normed_distance = normed_distance[sort_idx]
        
        plt.plot(
            time_history,
            normed_distance,
            label=f"{algo_name} ({len(time_history)} points)",
            color=color,
            linewidth=2,
            marker='.',
            markersize=4,
            alpha=0.8
        )
        
        # Print some statistics
        print(f"\n{algo_name} statistics:")
        print(f"Time range: {time_history[0]:.2f}s to {time_history[-1]:.2f}s")
        print(f"Norm range: {max(normed_distance):.2e} to {normed_distance[-1]:.2e}")
    
    # Customize the plot
    plt.xlabel('Runtime (seconds)', fontsize=12)
    plt.ylabel('Norm Distance', fontsize=12)
    plt.title('Convergence Comparison of SGD Algorithms (Convex)', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with nice formatting
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Check if we should use log scale
    all_norms = np.concatenate([data['normed_distance'] 
                              for data in algorithm_data.values()])
    if max(all_norms) / min(all_norms[all_norms > 0]) > 100:
        plt.yscale('log')
        print("\nUsing logarithmic scale for y-axis due to large range in norm values")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()

    
# Create the plot
plot_sgd_comparison(
    algorithms_data
)