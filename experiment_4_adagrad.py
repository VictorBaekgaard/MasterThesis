import generate_data, algorithms, loss_functions
import torch
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['figure.dpi'] = 150

def analyze_adagrad_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 10):
    """
    Analyze AdaGrad's convergence rate and variance reduction properties
    """
    # Problem parameters
    lambda_reg = 0.01
    func_args = {'lambda_reg': lambda_reg}

    # Generate problem instance
    X, y, theta_true = generate_data.generate_strongly_convex_data(
        n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=25.0, random_state=302
    )
    
    data = list(zip(X, y))
    
    # Estimate problem constants
    X_tensor = torch.stack([x for x, _ in data])
    eigenvalues = torch.linalg.eigvalsh(X_tensor.T @ X_tensor / n_samples)
    L_data = eigenvalues.max().item() + lambda_reg
    mu_data = eigenvalues.min().item() + lambda_reg
    condition_number = L_data / mu_data
    print(f"Data condition number: {condition_number:.2f}")

    # AdaGrad parameters
    base_lr = 0.15  # Initial learning rate η
    n_epochs = 2000
    total_iterations = n_epochs
    
    print(f"\nAdaGrad Configuration:")
    print(f"Base learning rate: {base_lr}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    distances_all = []
    losses_all = []
    param_trajectories_all = []
    
    for run in range(n_runs):
            # Generate problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302+run
        )
        
        data = list(zip(X, y))
        theta_init = torch.zeros_like(theta_true)
        
        # Run AdaGrad
        final_theta, loss_history, theta_history = algorithms.AdaGrad(
            func=loss_functions.ridge_regression_loss,
            lr=base_lr,
            n_iter=n_epochs,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args
        )
        
        # Compute distances from optimal
        distances = [torch.norm(theta - theta_true).item()**2 for theta in theta_history]
        distances_all.append(distances)
        losses_all.append(loss_history)
        param_trajectories_all.append(theta_history)
    
    # Convert to numpy arrays and compute statistics
    distances_all = np.array(distances_all)
    losses_all = np.array(losses_all)
    mean_distances = np.mean(distances_all, axis=0)
    variance_trajectory = np.var(distances_all, axis=0)
    mean_losses = np.mean(losses_all, axis=0)
    
    # Theoretical bounds
    t = np.arange(1, len(mean_distances) + 1)
    theoretical_variance = (base_lr**2 * L_data**2) / t  # From equation (adagrad.variance)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter convergence
    plt.subplot(1, 3, 1)
    plt.semilogy(mean_distances, 'b-', label='AdaGrad (Empirical)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||θ - θ*||²')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Variance reduction
    plt.subplot(1, 3, 2)
    plt.loglog(t, variance_trajectory, 'b-', label='Empirical Variance', linewidth=2)
    plt.loglog(t, theoretical_variance, 'r--', 
              label='Theoretical O(1/t) bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('Variance Reduction')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Loss convergence
    plt.subplot(1, 3, 3)
    plt.semilogy(mean_losses, 'b-', label='Mean Loss', linewidth=2)
    plt.fill_between(range(len(mean_losses)),
                    np.mean(losses_all, axis=0) - np.std(losses_all, axis=0),
                    np.mean(losses_all, axis=0) + np.std(losses_all, axis=0),
                    alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute and print statistics
    print("\nConvergence Analysis:")
    print(f"Initial learning rate: {base_lr:.4e}")
    
    # Verify variance reduction rate
    early_variance = variance_trajectory[len(variance_trajectory)//4]
    late_variance = variance_trajectory[-1]
    empirical_reduction = early_variance / late_variance
    theoretical_reduction = t[-1] / (len(variance_trajectory)//4)
    
    print("\nVariance Reduction Analysis:")
    print(f"Empirical variance reduction ratio: {empirical_reduction:.4f}")
    print(f"Theoretical reduction ratio (O(1/t)): {theoretical_reduction:.4f}")
    print(f"Final distance to optimum: {mean_distances[-1]:.4e}")
    print(f"Final loss value: {mean_losses[-1]:.4e}")
    
    return {
        'distances': mean_distances,
        'variance': variance_trajectory,
        'theoretical_variance': theoretical_variance,
        'losses': mean_losses,
    }

results = analyze_adagrad_convergence()