import algorithms, visualize, generate_data, loss_functions
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150

def momentum_variance_study(data, theta_true, loss_func,
                          momentum_lr_pairs, n_iter,
                          n_runs = 5, func_args = {}):
    """
    Experiment 1a: Impact of Momentum on Gradient Variance
    Tracks momentum vector variance during optimization
    """
    results = {}
    
    # Split data into x and y
    x_data = torch.stack([point[0] for point in data])
    y_data = torch.stack([point[1] for point in data])
    
    for beta, lr in momentum_lr_pairs:
        # Initialize storage for multiple runs
        all_distances = []
        all_momentum_vars = []  # Track variance of momentum vector
        
        for run in range(n_runs):
            # Different random initialization for each run
            theta_init = torch.zeros_like(theta_true)
            
            # Storage for this run's momentum vectors
            momentum_vectors = []
            
            # Run optimization and collect momentum vectors
            opt_theta, loss_history, theta_history, momentum_history = algorithms.MomentumSGD(
                func=loss_func,
                lr=lr,
                momentum=beta,
                n_iter=n_iter,
                data=data,
                theta_true=theta_true,
                batch_size=1,
                theta_init=theta_init,
                func_args=func_args,
                track_momentum=True  # Need to modify MomentumSGD to track v_t
            )
            
            # Calculate distances for this run
            normed_distance = [torch.norm(theta - theta_true).item() for theta in theta_history]
            
            all_distances.append(normed_distance)
            all_momentum_vars.append([torch.norm(v).item() for v in momentum_history])
        
        # Convert to numpy for easier averaging
        all_distances = np.array(all_distances)
        all_momentum_vars = np.array(all_momentum_vars)
        
        # Store average results and standard deviation
        results[beta] = {
            'distance_mean': np.mean(all_distances, axis=0),
            'distance_std': np.std(all_distances, axis=0),
            'momentum_var_mean': np.mean(all_momentum_vars, axis=0),
            'momentum_var_std': np.std(all_momentum_vars, axis=0)
        }
    
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameter Distance
    for beta, res in results.items():
        mean_dist = res['distance_mean']
        std_dist = res['distance_std']
        ax1.semilogy(mean_dist, label=f'β={beta}')
        ax1.fill_between(range(len(mean_dist)),
                        mean_dist - std_dist,
                        mean_dist + std_dist,
                        alpha=0.2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Distance from true parameter')
    ax1.set_title(f'Parameter Convergence over {n_runs} Runs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Gradient Variance
    for beta, res in results.items():
        mean_var = res['momentum_var_mean']
        std_var = res['momentum_var_std']
        ax2.semilogy(mean_var, label=f'β={beta}')
        ax2.fill_between(range(len(mean_var)),
                        mean_var - std_var,
                        mean_var + std_var,
                        alpha=0.2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Gradient Variance')
    ax2.set_title('Gradient Variance Evolution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results



# Generate data
X, y, theta_true = generate_data.generate_strongly_convex_data(
    n_samples=1000,
    n_features=5,
    noise_std=0.5,
    condition_number=10.0
)
data = list(zip(X, y))

momentum_lr_pairs = [
    (0.0, 0.01),    # Standard SGD can use larger lr
    (0.5, 0.005),   # Medium momentum needs smaller lr
    (0.9, 0.001),   # High momentum needs even smaller
    (0.99, 0.0001)  # Very high momentum needs tiny lr
]

# Set experiment parameters
n_iter = 2000
func_args = {'lambda_reg': 0.01}  # For ridge regression

# Run experiment
results = momentum_variance_study(
    data=data,
    theta_true=theta_true,
    loss_func=loss_functions.ridge_regression_loss,
    momentum_lr_pairs=momentum_lr_pairs,
    n_iter=n_iter,
    n_runs=5,
    func_args=func_args
)


####
def compare_momentum_vs_minibatch(data, theta_true, loss_func,
                                momentum_params, minibatch_params, 
                                n_iter, n_runs=5, func_args={}):
    """
    Compare one momentum configuration with one mini-batch configuration
    """
    results = {'momentum': {}, 'minibatch': {}}
    
    # Split data into x and y
    x_data = torch.stack([point[0] for point in data])
    y_data = torch.stack([point[1] for point in data])
    
    # Unpack parameters
    momentum_beta, momentum_lr = momentum_params
    batch_size, minibatch_lr = minibatch_params
    
    for run in range(n_runs):
        # Different random initialization for each run
        theta_init = torch.zeros_like(theta_true)
        
        # Run Momentum SGD
        _, momentum_loss, momentum_theta_hist, momentum_vectors = algorithms.MomentumSGD(
            func=loss_func,
            lr=momentum_lr,
            momentum=momentum_beta,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            batch_size=1,  # Fixed batch size for momentum
            theta_init=theta_init.clone(),
            func_args=func_args,
            track_momentum=True
        )
        
        # Run Mini-batch SGD
        _, minibatch_loss, minibatch_theta_hist = algorithms.SGD(
            func=loss_func,
            lr=minibatch_lr,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            batch_size=batch_size,
            theta_init=theta_init.clone(),
            func_args=func_args,
            decay_rate=0.0
        )
        
        # Calculate distances
        momentum_dist = [torch.norm(theta - theta_true).item() 
                        for theta in momentum_theta_hist]
        minibatch_dist = [torch.norm(theta - theta_true).item() 
                         for theta in minibatch_theta_hist]
        
        # Store results for this run
        if run == 0:
            results['momentum']['distances'] = [momentum_dist]
            results['momentum']['losses'] = [momentum_loss]
            results['momentum']['momentum_norms'] = [momentum_vectors]
            
            results['minibatch']['distances'] = [minibatch_dist]
            results['minibatch']['losses'] = [minibatch_loss]
        else:
            results['momentum']['distances'].append(momentum_dist)
            results['momentum']['losses'].append(momentum_loss)
            results['momentum']['momentum_norms'].append(momentum_vectors)
            
            results['minibatch']['distances'].append(minibatch_dist)
            results['minibatch']['losses'].append(minibatch_loss)
    
    # Convert to numpy and compute statistics
    for method in ['momentum', 'minibatch']:
        for metric in ['distances', 'losses']:
            data = np.array(results[method][metric])
            results[method][f'{metric}_mean'] = np.mean(data, axis=0)
            results[method][f'{metric}_std'] = np.std(data, axis=0)
            
        if method == 'momentum':
            momentum_norms = np.array(results[method]['momentum_norms'])
            results[method]['momentum_norms_mean'] = np.mean(momentum_norms, axis=0)
            results[method]['momentum_norms_std'] = np.std(momentum_norms, axis=0)
    
    # Plotting
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameter Distance
    for method, label, color in [('momentum', f'Momentum (β={momentum_beta})', 'blue'),
                                ('minibatch', f'Mini-batch (size={batch_size})', 'red')]:
        mean_dist = results[method]['distances_mean']
        std_dist = results[method]['distances_std']
        plt.semilogy(mean_dist, label=label, color=color)
        plt.fill_between(range(len(mean_dist)),
                        mean_dist - std_dist,
                        mean_dist + std_dist,
                        alpha=0.2,
                        color=color)
    
    plt.xlabel('Iterations')
    plt.ylabel('Distance from true parameter')
    plt.title(f'Parameter Convergence over {n_runs} Runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results




# Example usage:
data, theta_true = generate_data.generate_convex_data(
    N=2000,
    d=5,
    random_state=30221
)


# Parameters for both methods
momentum_params = (0.75, 0.001)  # (beta, lr)
minibatch_params = (1, 0.01)   # (batch_size, lr)
n_iter = 2000
#func_args = {'lambda_reg': 0.01}

# Run comparison
results = compare_momentum_vs_minibatch(
    data=data,
    theta_true=theta_true,
    loss_func=loss_functions.absolute_loss,
    momentum_params=momentum_params,
    minibatch_params=minibatch_params,
    n_iter=n_iter,
    n_runs=10
)