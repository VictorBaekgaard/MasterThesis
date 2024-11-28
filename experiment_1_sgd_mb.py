import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import numpy as np
import algorithms, loss_functions, generate_data, visualize

plt.rcParams['figure.dpi'] = 150

def measure_gradient_variance(loss_func, data: torch.Tensor, theta: torch.Tensor, 
                            batch_sizes: List[int], num_estimates: int = 100,
                            func_args: Dict = {}) -> Dict[int, float]:
    """
    Measure gradient variance for different batch sizes at a fixed point
    """
    variances = {}
    device = theta.device
    
    # Create a copy of theta that requires gradient
    theta = theta.clone().detach().requires_grad_(True)
    
    # Split data into x and y
    x_data = torch.stack([point[0] for point in data])
    y_data = torch.stack([point[1] for point in data])
    
    # Compute true gradient (full batch)
    true_grads = []
    for x_i, y_i in zip(x_data, y_data):
        loss = loss_func(theta, x_i, y_i, **func_args)
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        true_grads.append(grad)
    true_grad = torch.mean(torch.stack(true_grads), dim=0)
    
    for batch_size in batch_sizes:
        gradient_estimates = []
        for _ in range(num_estimates):
            # Randomly sample batch
            indices = torch.randperm(len(data))[:batch_size]
            batch_x = x_data[indices]
            batch_y = y_data[indices]
            
            # Compute batch gradient
            batch_grads = []
            for x_i, y_i in zip(batch_x, batch_y):
                loss = loss_func(theta, x_i, y_i, **func_args)
                grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
                batch_grads.append(grad)
            
            batch_grad = torch.mean(torch.stack(batch_grads), dim=0)
            gradient_estimates.append(batch_grad)
        
        # Compute empirical variance
        gradient_estimates = torch.stack(gradient_estimates)
        variance = torch.mean(torch.norm(gradient_estimates - true_grad, dim=1)**2)
        variances[batch_size] = variance.item()
        
    return variances

def run_variance_analysis(data: torch.Tensor, theta_true: torch.Tensor, loss_func,
                         batch_sizes: List[int], func_args: Dict = {}):
    """
    Experiment 1.1: Gradient Variance Measurement
    """
    # Measure variance at different points along optimization path
    points_to_measure = [
        torch.randn_like(theta_true),  # Random initial point
        theta_true + 0.1 * torch.randn_like(theta_true),  # Near optimum
        theta_true.clone()  # At optimum
    ]
    
    all_variances = []
    for point in points_to_measure:
        variances = measure_gradient_variance(loss_func, data, point, batch_sizes, func_args=func_args)
        all_variances.append(variances)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for i, variances in enumerate(all_variances):
        batch_sizes_list = list(variances.keys())
        variances_list = list(variances.values())
        plt.loglog(batch_sizes_list, variances_list, 'o-', 
                  label=f'Point {i+1}')
    
    # Plot theoretical O(1/B) line
    ref_x = torch.tensor(batch_sizes)
    ref_y = variances_list[0] * (batch_sizes[0] / torch.tensor(ref_x).float())
    plt.loglog(ref_x, ref_y, '--', label='O(1/B)')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Gradient Variance')
    plt.title('Gradient Variance vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.show()

def batch_size_impact_study(data: torch.Tensor, theta_true: torch.Tensor, loss_func,
                          batch_sizes: List[int], lr: float, n_iter: int,
                          n_runs: int = 5, func_args: Dict = {}):
    """
    Experiment 1.2: Impact of Batch Size
    Averages results over multiple runs to reduce variance
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Initialize storage for multiple runs
        all_distances = []
        all_thetas = []
        
        for run in range(n_runs):
            # Different random initialization for each run
            theta_init = torch.rand_like(theta_true)
            
            # Run optimization
            opt_theta, loss_history, theta_history, _ = algorithms.SGD(
                func=loss_func,
                lr=lr,
                n_iter=n_iter,
                data=data,
                theta_true=theta_true,
                batch_size=batch_size,
                decay_rate=0.0,
                theta_init=theta_init,
                func_args=func_args,
                do_time=True
            )
            
            # Calculate distances for this run
            normed_distance = [torch.norm(theta - theta_true).item() for theta in theta_history]
            all_distances.append(normed_distance)
            all_thetas.append(theta_history)
        
        # Convert to numpy for easier averaging
        all_distances = np.array(all_distances)
        
        # Store average results and standard deviation
        results[batch_size] = {
            'loss_history_mean': np.mean(all_distances, axis=0),
            'loss_history_std': np.std(all_distances, axis=0),
            'theta_histories': all_thetas  # Keep all theta histories if needed
        }
    
    # Plot average convergence for different batch sizes with error bands
    plt.figure(figsize=(10, 6))
    for batch_size, res in results.items():
        mean_loss = res['loss_history_mean']
        std_loss = res['loss_history_std']
        
        # Plot mean line
        plt.semilogy(mean_loss, label=f'B={batch_size}')
        
        # Add error bands (±1 standard deviation)
        plt.fill_between(range(len(mean_loss)),
                        mean_loss - std_loss,
                        mean_loss + std_loss,
                        alpha=0.2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Distance from true parameter')
    plt.title(f'Average Convergence over {n_runs} Runs for Different Batch Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

def convex_convergence_analysis(n_samples: int = 10000, d: int = 2, n_runs: int = 5):
    """
    Experiment 2.1: Convex Problems
    Averages results over multiple runs to reduce variance
    
    Args:
        n_samples: Number of data samples
        d: Dimension of the problem
        n_runs: Number of runs to average over (default: 3)
    """
    # Parameters
    lr = 0.01  # Should be <= 1/L
    n_iter = 1000
    
    # Storage for multiple runs
    sgd_distances_all = []
    mbsgd_distances_all = []
    
    for run in range(n_runs):
        # Generate new convex problem for each run
        data, theta_true = generate_data.generate_convex_data(n_samples, d)
        theta_init = torch.rand_like(theta_true)
        
        # Run SGD and MB-SGD
        sgd_results = algorithms.SGD(
            func=loss_functions.absolute_loss,
            lr=lr,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            batch_size=1,
            decay_rate=0.0,
            theta_init=theta_init
        )
        
        mbsgd_results = algorithms.SGD(
            func=loss_functions.absolute_loss,
            lr=lr,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            decay_rate=0.0,
            theta_init=theta_init,
            batch_size=32
        )
        
        # Calculate distances for this run
        sgd_distances = [torch.norm(theta - theta_true).item() for theta in sgd_results[2]]
        mbsgd_distances = [torch.norm(theta - theta_true).item() for theta in mbsgd_results[2]]
        
        sgd_distances_all.append(sgd_distances)
        mbsgd_distances_all.append(mbsgd_distances)
    
    # Convert to numpy arrays for easier computation
    sgd_distances_all = np.array(sgd_distances_all)
    mbsgd_distances_all = np.array(mbsgd_distances_all)
    
    # Calculate means and standard deviations
    sgd_mean = np.mean(sgd_distances_all, axis=0)
    sgd_std = np.std(sgd_distances_all, axis=0)
    mbsgd_mean = np.mean(mbsgd_distances_all, axis=0)
    mbsgd_std = np.std(mbsgd_distances_all, axis=0)
    
    # Plot average convergence comparison
    plt.figure(figsize=(10, 6))
    
    # Plot SGD
    plt.semilogy(sgd_mean, label='SGD (batch size 1)', color='blue')
    plt.fill_between(range(len(sgd_mean)),
                    sgd_mean - sgd_std,
                    sgd_mean + sgd_std,
                    color='blue', alpha=0.2)
    
    # Plot MB-SGD
    plt.semilogy(mbsgd_mean, label='MB-SGD (batch size 32)', color='red')
    plt.fill_between(range(len(mbsgd_mean)),
                    mbsgd_mean - mbsgd_std,
                    mbsgd_mean + mbsgd_std,
                    color='red', alpha=0.2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Distance from true parameter')
    plt.title(f'Average Convergence Comparison over {n_runs} Runs (Convex Problem)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize optimization paths
    visualize.visualize_optimization(data, sgd_results[2], theta_true, loss_functions.absolute_loss,
    "SGD Path (convex)")
    visualize.visualize_optimization(data, mbsgd_results[2], theta_true, loss_functions.absolute_loss,
    "MB-SGD Path (convex)")

    return {
        'sgd': {
            'mean': sgd_mean,
            'std': sgd_std,
            'all_runs': sgd_distances_all
        },
        'mbsgd': {
            'mean': mbsgd_mean,
            'std': mbsgd_std,
            'all_runs': mbsgd_distances_all
        }
    }

def strongly_convex_convergence_analysis(n_samples: int = 10000, d: int = 2, n_runs: int = 5):
    """
    Experiment 2.1: Convex Problems
    Averages results over multiple runs to reduce variance
    
    Args:
        n_samples: Number of data samples
        d: Dimension of the problem
        n_runs: Number of runs to average over (default: 3)
    """
    # Parameters
    lr = 0.01  # Should be <= 1/L
    n_iter = 1000
    
    # Storage for multiple runs
    sgd_distances_all = []
    mbsgd_distances_all = []
    
    lambda_reg = 0.1
    func_args = {'lambda_reg': lambda_reg}

    for run in range(n_runs):
        # Generate new convex problem for each run
        X, y, theta_true = generate_data.generate_strongly_convex_data(n_samples=1000, n_features=2, noise_std=0.1, random_state=302)
        data = list(zip(X, y))

        theta_init = torch.rand_like(theta_true)
        
        # Run SGD and MB-SGD
        sgd_results = algorithms.SGD(
            func=loss_functions.ridge_regression_loss,
            lr=lr,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            batch_size=1,
            decay_rate=0.0,
            theta_init=theta_init,
            func_args=func_args
        )
        
        mbsgd_results = algorithms.SGD(
            func=loss_functions.ridge_regression_loss,
            lr=lr,
            n_iter=n_iter,
            data=data,
            theta_true=theta_true,
            decay_rate=0.0,
            theta_init=theta_init,
            batch_size=32,
            func_args=func_args
        )
        
        # Calculate distances for this run
        sgd_distances = [torch.norm(theta - theta_true).item() for theta in sgd_results[2]]
        mbsgd_distances = [torch.norm(theta - theta_true).item() for theta in mbsgd_results[2]]
        
        sgd_distances_all.append(sgd_distances)
        mbsgd_distances_all.append(mbsgd_distances)
    
    # Convert to numpy arrays for easier computation
    sgd_distances_all = np.array(sgd_distances_all)
    mbsgd_distances_all = np.array(mbsgd_distances_all)
    
    # Calculate means and standard deviations
    sgd_mean = np.mean(sgd_distances_all, axis=0)
    sgd_std = np.std(sgd_distances_all, axis=0)
    mbsgd_mean = np.mean(mbsgd_distances_all, axis=0)
    mbsgd_std = np.std(mbsgd_distances_all, axis=0)
    
    # Plot average convergence comparison
    plt.figure(figsize=(10, 6))
    
    # Plot SGD
    plt.semilogy(sgd_mean, label='SGD (batch size 1)', color='blue')
    plt.fill_between(range(len(sgd_mean)),
                    sgd_mean - sgd_std,
                    sgd_mean + sgd_std,
                    color='blue', alpha=0.2)
    
    # Plot MB-SGD
    plt.semilogy(mbsgd_mean, label='MB-SGD (batch size 32)', color='red')
    plt.fill_between(range(len(mbsgd_mean)),
                    mbsgd_mean - mbsgd_std,
                    mbsgd_mean + mbsgd_std,
                    color='red', alpha=0.2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Distance from true parameter')
    plt.title(f'Average Convergence Comparison over {n_runs} Runs (Strongly Convex Problem)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize optimization paths
    visualize.visualize_optimization(data, sgd_results[2], theta_true, loss_functions.ridge_regression_loss,
    "SGD Path (strongly convex)", func_args=func_args)
    visualize.visualize_optimization(data, mbsgd_results[2], theta_true, loss_functions.ridge_regression_loss,
    "MB-SGD Path (strongly convex)", func_args=func_args)

    return {
        'sgd': {
            'mean': sgd_mean,
            'std': sgd_std,
            'all_runs': sgd_distances_all
        },
        'mbsgd': {
            'mean': mbsgd_mean,
            'std': mbsgd_std,
            'all_runs': mbsgd_distances_all
        }
    }


def run_all_experiments():
    # Common parameters
    n_samples = 10000
    d = 2
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    lr = 0.01
    n_iter = 300
    
    # Generate datasets
    convex_data, convex_theta_true = generate_data.generate_convex_data(n_samples, d)
    #strongly_convex_data, sc_theta_true, A, b = generate_data.generate_quadratic_data(n_samples, d, condition_number=5.0)


    # Experiment 1.1: Variance Analysis
    print("Running Variance Analysis...")
    run_variance_analysis(convex_data, convex_theta_true, loss_functions.absolute_loss, batch_sizes)
    
    # Experiment 1.2: Batch Size Impact
    print("Running Batch Size Impact Study...")
    batch_size_impact_study(convex_data, convex_theta_true, loss_functions.absolute_loss, 
                          batch_sizes, lr, n_iter)
    
    # Experiment 2.1: Convex Convergence
    print("Running Convex Convergence Analysis...")
    convex_convergence_analysis()
    # Experiment 2.1: Convex Convergence
    print("Running Strongly Convex Convergence Analysis...")
    strongly_convex_convergence_analysis()

if __name__ == "__main__":
    run_all_experiments()