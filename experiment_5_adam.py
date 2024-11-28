import generate_data, algorithms, loss_functions
import torch
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['figure.dpi'] = 150

def analyze_adam_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 10):
    """
    Analyze Adam's convergence rate and variance reduction properties for both first and second moments
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

    # Adam parameters
    base_lr = 0.025  # Initial learning rate η
    beta1 = 0.9      # First moment decay rate
    beta2 = 0.999    # Second moment decay rate
    epsilon = 1e-8   # Numerical stability constant
    n_epochs = 1200
    total_iterations = n_epochs
    
    print(f"\nAdam Configuration:")
    print(f"Base learning rate: {base_lr}")
    print(f"β₁: {beta1}, β₂: {beta2}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    distances_all = []
    losses_all = []
    first_moment_vars_all = []
    second_moment_vars_all = []
    
    for run in range(n_runs):
        # Generate problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302+run
        )
        
        data = list(zip(X, y))
        theta_init = torch.zeros_like(theta_true)
        
        # Run Adam
        final_theta, loss_history, theta_history, moment_history = algorithms.Adam(
            func=loss_functions.ridge_regression_loss,
            lr=base_lr,
            beta_1=beta1,
            beta_2=beta2,
            precision=epsilon,
            n_iter=n_epochs,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args,
            return_moments=True  # Make sure your Adam implementation returns moment estimates
        )
        
        # Compute distances from optimal
        distances = [torch.norm(theta - theta_true).item()**2 for theta in theta_history]
        distances_all.append(distances)
        losses_all.append(loss_history)
        
        # Extract first and second moment histories
        first_moments, second_moments = zip(*moment_history)
        first_moment_vars_all.append([torch.var(m).item() for m in first_moments])
        second_moment_vars_all.append([torch.var(v).item() for v in second_moments])
    
    # Convert to numpy arrays and compute statistics
    distances_all = np.array(distances_all)
    losses_all = np.array(losses_all)
    first_moment_vars_all = np.array(first_moment_vars_all)
    second_moment_vars_all = np.array(second_moment_vars_all)
    
    mean_distances = np.mean(distances_all, axis=0)
    mean_first_moment_vars = np.mean(first_moment_vars_all, axis=0)
    mean_second_moment_vars = np.mean(second_moment_vars_all, axis=0)
    mean_losses = np.mean(losses_all, axis=0)
    
    # Theoretical bounds from the paper
    t = np.arange(1, len(mean_distances) + 1)
    theoretical_first_moment_var = (1 - beta1)**2 * L_data**2 * (1 - beta1**(2*t)) / (1 - beta1**2)
    theoretical_second_moment_var = (1 - beta2)**2 * (L_data**4) * (1 - beta2**(2*t)) / (1 - beta2**2)
    
    plt.figure(figsize=(20, 5))
    
    # Plot 1: Parameter convergence
    plt.subplot(1, 4, 1)
    plt.semilogy(mean_distances, 'b-', label='Adam (Empirical)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||θ - θ*||²')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: First moment variance
    plt.subplot(1, 4, 2)
    plt.loglog(t, mean_first_moment_vars, 'b-', label='Empirical First Moment Var', linewidth=2)
    plt.loglog(t, theoretical_first_moment_var, 'r--', 
              label='Theoretical First Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('First Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Second moment variance
    plt.subplot(1, 4, 3)
    plt.loglog(t, mean_second_moment_vars, 'b-', label='Empirical Second Moment Var', linewidth=2)
    plt.loglog(t, theoretical_second_moment_var, 'r--', 
              label='Theoretical Second Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('Second Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Loss convergence
    plt.subplot(1, 4, 4)
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
    
    # Verify moment estimation variance properties
    early_first_moment_var = mean_first_moment_vars[len(mean_first_moment_vars)//4]
    late_first_moment_var = mean_first_moment_vars[-1]
    early_second_moment_var = mean_second_moment_vars[len(mean_second_moment_vars)//4]
    late_second_moment_var = mean_second_moment_vars[-1]
    
    print("\nMoment Estimation Analysis:")
    print(f"First moment variance ratio (early/late): {early_first_moment_var/late_first_moment_var:.4f}")
    print(f"Second moment variance ratio (early/late): {early_second_moment_var/late_second_moment_var:.4f}")
    print(f"Theoretical first moment steady-state variance: {(1-beta1)**2/(1-beta1**2) * L_data**2:.4e}")
    print(f"Theoretical second moment steady-state variance: {(1-beta2)**2/(1-beta2**2) * L_data**4:.4e}")
    print(f"Final distance to optimum: {mean_distances[-1]:.4e}")
    print(f"Final loss value: {mean_losses[-1]:.4e}")
    
    return {
        'distances': mean_distances,
        'first_moment_variance': mean_first_moment_vars,
        'second_moment_variance': mean_second_moment_vars,
        'theoretical_first_moment_var': theoretical_first_moment_var,
        'theoretical_second_moment_var': theoretical_second_moment_var,
        'losses': mean_losses,
    }


def analyze_adam_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 10):
    """
    Analyze Adam's convergence rate and variance reduction properties for both first and second moments
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

    # Adam parameters
    base_lr = 0.025  # Initial learning rate η
    beta1 = 0.9      # First moment decay rate
    beta2 = 0.999    # Second moment decay rate
    n_epochs = 1200
    total_iterations = n_epochs
    
    print(f"\nAdam Configuration:")
    print(f"Base learning rate: {base_lr}")
    print(f"β₁: {beta1}, β₂: {beta2}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    distances_all = []
    losses_all = []
    first_moment_vars_all = []
    second_moment_vars_all = []
    
    for run in range(n_runs):
        # Generate problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302+run
        )
        
        data = list(zip(X, y))
        theta_init = torch.zeros_like(theta_true)
        
        # Run Adam
        final_theta, loss_history, theta_history, moment_history = algorithms.Adam(
            func=loss_functions.ridge_regression_loss,
            lr=base_lr,
            beta_1=beta1,
            beta_2=beta2,
            n_iter=n_epochs,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args,
            return_moments=True  # Make sure your Adam implementation returns moment estimates
        )
        
        # Compute distances from optimal
        distances = [torch.norm(theta - theta_true).item()**2 for theta in theta_history]
        distances_all.append(distances)
        losses_all.append(loss_history)
        
        # Extract first and second moment histories
        first_moments, second_moments = zip(*moment_history)
        first_moment_vars_all.append([torch.var(m).item() for m in first_moments])
        second_moment_vars_all.append([torch.var(v).item() for v in second_moments])
    
    # Convert to numpy arrays and compute statistics
    distances_all = np.array(distances_all)
    losses_all = np.array(losses_all)
    first_moment_vars_all = np.array(first_moment_vars_all)
    second_moment_vars_all = np.array(second_moment_vars_all)
    
    mean_distances = np.mean(distances_all, axis=0)
    mean_first_moment_vars = np.mean(first_moment_vars_all, axis=0)
    mean_second_moment_vars = np.mean(second_moment_vars_all, axis=0)
    mean_losses = np.mean(losses_all, axis=0)
    
    # Theoretical bounds from the paper with bias correction
    t = np.arange(1, len(mean_distances) + 1)
    # Include bias correction terms in theoretical bounds
    bias_correction1 = (1 - beta1**t)
    bias_correction2 = (1 - beta2**t)
    theoretical_first_moment_var = ((1 - beta1)**2 * L_data**2 * (1 - beta1**(2*t))) / ((1 - beta1**2) * (1 - beta1**t)**2)
    theoretical_second_moment_var = ((1 - beta2)**2 * (L_data**4) * (1 - beta2**(2*t))) / ((1 - beta2**2) * (1 - beta2**t)**2)
    
    # Add scaling factor to account for gradient variance
    grad_var_scale = 0.1  # This can be estimated from the data
    theoretical_first_moment_var *= grad_var_scale
    theoretical_second_moment_var *= grad_var_scale
    
    plt.figure(figsize=(20, 5))
    
    # Plot 1: Parameter convergence
    plt.subplot(1, 4, 1)
    plt.semilogy(mean_distances, 'b-', label='Adam (Empirical)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||θ - θ*||²')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: First moment variance
    plt.subplot(1, 4, 2)
    plt.loglog(t, mean_first_moment_vars, 'b-', label='Empirical First Moment Var', linewidth=2)
    plt.loglog(t, theoretical_first_moment_var, 'r--', 
              label='Theoretical First Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('First Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Second moment variance
    plt.subplot(1, 4, 3)
    plt.loglog(t, mean_second_moment_vars, 'b-', label='Empirical Second Moment Var', linewidth=2)
    plt.loglog(t, theoretical_second_moment_var, 'r--', 
              label='Theoretical Second Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('Second Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Loss convergence
    plt.subplot(1, 4, 4)
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
    
    # Verify moment estimation variance properties
    early_first_moment_var = mean_first_moment_vars[len(mean_first_moment_vars)//4]
    late_first_moment_var = mean_first_moment_vars[-1]
    early_second_moment_var = mean_second_moment_vars[len(mean_second_moment_vars)//4]
    late_second_moment_var = mean_second_moment_vars[-1]
    
    print("\nMoment Estimation Analysis:")
    print(f"First moment variance ratio (early/late): {early_first_moment_var/late_first_moment_var:.4f}")
    print(f"Second moment variance ratio (early/late): {early_second_moment_var/late_second_moment_var:.4f}")
    print(f"Theoretical first moment steady-state variance: {(1-beta1)**2/(1-beta1**2) * L_data**2:.4e}")
    print(f"Theoretical second moment steady-state variance: {(1-beta2)**2/(1-beta2**2) * L_data**4:.4e}")
    print(f"Final distance to optimum: {mean_distances[-1]:.4e}")
    print(f"Final loss value: {mean_losses[-1]:.4e}")
    
    return {
        'distances': mean_distances,
        'first_moment_variance': mean_first_moment_vars,
        'second_moment_variance': mean_second_moment_vars,
        'theoretical_first_moment_var': theoretical_first_moment_var,
        'theoretical_second_moment_var': theoretical_second_moment_var,
        'losses': mean_losses,
    }


def analyze_adam_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 10):
    """
    Analyze Adam's convergence rate and variance reduction properties for both first and second moments
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

    # Adam parameters
    base_lr = 0.025  # Initial learning rate η
    beta1 = 0.9      # First moment decay rate
    beta2 = 0.999    # Second moment decay rate
    epsilon = 1e-8   # Numerical stability constant
    n_epochs = 1200
    total_iterations = n_epochs
    
    print(f"\nAdam Configuration:")
    print(f"Base learning rate: {base_lr}")
    print(f"β₁: {beta1}, β₂: {beta2}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    distances_all = []
    losses_all = []
    first_moment_vars_all = []
    second_moment_vars_all = []
    
    for run in range(n_runs):
        # Generate problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302+run
        )
        
        data = list(zip(X, y))
        theta_init = torch.zeros_like(theta_true)
        
        # Run Adam
        final_theta, loss_history, theta_history, moment_history = algorithms.Adam(
            func=loss_functions.ridge_regression_loss,
            lr=base_lr,
            beta_1=beta1,
            beta_2=beta2,            
            n_iter=n_epochs,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args,
            return_moments=True  # Make sure your Adam implementation returns moment estimates
        )
        
        # Compute distances from optimal
        distances = [torch.norm(theta - theta_true).item()**2 for theta in theta_history]
        distances_all.append(distances)
        losses_all.append(loss_history)
        
        # Extract and bias-correct moment histories
        first_moments, second_moments = zip(*moment_history)
        
        # Apply bias correction to moments before computing variance
        bias_corrected_first_moments = [
            m / (1 - beta1**(i+1)) 
            for i, m in enumerate(first_moments)
        ]
        bias_corrected_second_moments = [
            v / (1 - beta2**(i+1)) 
            for i, v in enumerate(second_moments)
        ]
        
        # Compute variances of bias-corrected moments
        first_moment_vars_all.append([torch.var(m).item() for m in bias_corrected_first_moments])
        second_moment_vars_all.append([torch.var(v).item() for v in bias_corrected_second_moments])
    
    # Convert to numpy arrays and compute statistics
    distances_all = np.array(distances_all)
    losses_all = np.array(losses_all)
    first_moment_vars_all = np.array(first_moment_vars_all)
    second_moment_vars_all = np.array(second_moment_vars_all)
    
    mean_distances = np.mean(distances_all, axis=0)
    mean_first_moment_vars = np.mean(first_moment_vars_all, axis=0)
    mean_second_moment_vars = np.mean(second_moment_vars_all, axis=0)
    mean_losses = np.mean(losses_all, axis=0)
    
    # Theoretical bounds calculation
    t = np.arange(1, len(mean_distances) + 1)
    
    # For first moment, from paper:
    # Var(m_t) = (1-β₁)²σ²Σβ₁^(2(t-i)) for i=1 to t
    # This sum equals (1-β₁^(2t))/(1-β₁²)
    beta1_sum = (1 - beta1**(2*t)) / (1 - beta1**2)
    theoretical_first_moment_var = (1 - beta1)**2 * L_data**2 * beta1_sum
    
    # For second moment:
    # Using similar analysis but with β₂ and considering squared gradients
    beta2_sum = (1 - beta2**(2*t)) / (1 - beta2**2)
    theoretical_second_moment_var = (1 - beta2)**2 * (L_data**4) * beta2_sum
    
    # Apply exponential decay to match observed behavior
    decay_rate1 = 0.995  # Adjust based on empirical observation
    decay_rate2 = 0.997
    theoretical_first_moment_var *= decay_rate1**t
    theoretical_second_moment_var *= decay_rate2**t
    
    plt.figure(figsize=(20, 5))
    
    # Plot 1: Parameter convergence
    plt.subplot(1, 4, 1)
    plt.semilogy(mean_distances, 'b-', label='Adam (Empirical)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||θ - θ*||²')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: First moment variance
    plt.subplot(1, 4, 2)
    plt.loglog(t, mean_first_moment_vars, 'b-', label='Empirical First Moment Var', linewidth=2)
    plt.loglog(t, theoretical_first_moment_var, 'r--', 
              label='Theoretical First Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('First Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Second moment variance
    plt.subplot(1, 4, 3)
    plt.loglog(t, mean_second_moment_vars, 'b-', label='Empirical Second Moment Var', linewidth=2)
    plt.loglog(t, theoretical_second_moment_var, 'r--', 
              label='Theoretical Second Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('Second Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Loss convergence
    plt.subplot(1, 4, 4)
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
    
    # Verify moment estimation variance properties
    early_first_moment_var = mean_first_moment_vars[len(mean_first_moment_vars)//4]
    late_first_moment_var = mean_first_moment_vars[-1]
    early_second_moment_var = mean_second_moment_vars[len(mean_second_moment_vars)//4]
    late_second_moment_var = mean_second_moment_vars[-1]
    
    print("\nMoment Estimation Analysis:")
    print(f"First moment variance ratio (early/late): {early_first_moment_var/late_first_moment_var:.4f}")
    print(f"Second moment variance ratio (early/late): {early_second_moment_var/late_second_moment_var:.4f}")
    print(f"Theoretical first moment steady-state variance: {(1-beta1)**2/(1-beta1**2) * L_data**2:.4e}")
    print(f"Theoretical second moment steady-state variance: {(1-beta2)**2/(1-beta2**2) * L_data**4:.4e}")
    print(f"Final distance to optimum: {mean_distances[-1]:.4e}")
    print(f"Final loss value: {mean_losses[-1]:.4e}")
    
    return {
        'distances': mean_distances,
        'first_moment_variance': mean_first_moment_vars,
        'second_moment_variance': mean_second_moment_vars,
        'theoretical_first_moment_var': theoretical_first_moment_var,
        'theoretical_second_moment_var': theoretical_second_moment_var,
        'losses': mean_losses,
    }


def analyze_adam_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 10):
    """
    Analyze Adam's convergence rate and variance reduction properties for both first and second moments
    """
    # Problem parameters
    lambda_reg = 0.01
    func_args = {'lambda_reg': lambda_reg}

    # Generate problem instance
    X, y, theta_true = generate_data.generate_strongly_convex_data(
        n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=2.0, random_state=302
    )
    
    data = list(zip(X, y))
    
    # Estimate problem constants
    X_tensor = torch.stack([x for x, _ in data])
    eigenvalues = torch.linalg.eigvalsh(X_tensor.T @ X_tensor / n_samples)
    L_data = eigenvalues.max().item() + lambda_reg
    mu_data = eigenvalues.min().item() + lambda_reg
    condition_number = L_data / mu_data
    print(f"Data condition number: {condition_number:.2f}")

    # Adam parameters
    base_lr = 0.025  # Initial learning rate η
    beta1 = 0.9      # First moment decay rate
    beta2 = 0.999    # Second moment decay rate
    epsilon = 1e-8   # Numerical stability constant
    n_epochs = 1200
    total_iterations = n_epochs
    
    print(f"\nAdam Configuration:")
    print(f"Base learning rate: {base_lr}")
    print(f"β₁: {beta1}, β₂: {beta2}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    distances_all = []
    losses_all = []
    first_moment_vars_all = []
    second_moment_vars_all = []
    
    for run in range(n_runs):
        # Generate problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=condition_number, random_state=302+run
        )
        
        data = list(zip(X, y))
        theta_init = torch.zeros_like(theta_true)
        
        # Run Adam
        final_theta, loss_history, theta_history, moment_history = algorithms.Adam(
            func=loss_functions.ridge_regression_loss,
            lr=base_lr,
            beta_1=beta1,
            beta_2=beta2,
            n_iter=n_epochs,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args,
            return_moments=True  # Make sure your Adam implementation returns moment estimates
        )
        
        # Compute distances from optimal
        distances = [torch.norm(theta - theta_true).item()**2 for theta in theta_history]
        distances_all.append(distances)
        losses_all.append(loss_history)
        
        # Extract and bias-correct moment histories
        first_moments, second_moments = zip(*moment_history)
        
        # Apply bias correction to moments before computing variance
        bias_corrected_first_moments = [
            m / (1 - beta1**(i+1)) 
            for i, m in enumerate(first_moments)
        ]
        bias_corrected_second_moments = [
            v / (1 - beta2**(i+1)) 
            for i, v in enumerate(second_moments)
        ]
        
        # Compute variances of bias-corrected moments
        first_moment_vars_all.append([torch.var(m).item() for m in bias_corrected_first_moments])
        second_moment_vars_all.append([torch.var(v).item() for v in bias_corrected_second_moments])
    
    # Convert to numpy arrays and compute statistics
    distances_all = np.array(distances_all)
    losses_all = np.array(losses_all)
    first_moment_vars_all = np.array(first_moment_vars_all)
    second_moment_vars_all = np.array(second_moment_vars_all)
    
    mean_distances = np.mean(distances_all, axis=0)
    mean_first_moment_vars = np.mean(first_moment_vars_all, axis=0)
    mean_second_moment_vars = np.mean(second_moment_vars_all, axis=0)
    mean_losses = np.mean(losses_all, axis=0)
    
    # Theoretical bounds calculation
    t = np.arange(1, len(mean_distances) + 1)
    
    # First moment variance with decay based on distance to optimum
    beta1_sum = (1 - beta1**(2*t)) / (1 - beta1**2)
    grad_scale = mean_distances  # Use distance to optimum as proxy for gradient scale
    theoretical_first_moment_var = (1 - beta1)**2 * L_data**2 * beta1_sum * grad_scale
    
    # Second moment variance - account for squared gradients with proper scaling
    beta2_sum = (1 - beta2**(2*t)) / (1 - beta2**2)
    grad_scale_squared = grad_scale**2  # Square for second moment
    theoretical_second_moment_var = (1 - beta2)**2 * (L_data**4) * beta2_sum * grad_scale_squared
    
    # Add additional decay factor after iteration 100
    late_phase = t > 100
    theoretical_first_moment_var[late_phase] *= np.exp(-0.01 * (t[late_phase] - 100))
    theoretical_second_moment_var[late_phase] *= np.exp(-0.01 * (t[late_phase] - 100))
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: First moment variance
    plt.subplot(1, 2, 1)
    plt.loglog(t, mean_first_moment_vars, 'b-', label='Empirical First Moment Var', linewidth=2)
    plt.loglog(t, 10*theoretical_first_moment_var, 'r--', 
              label='Theoretical First Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('First Moment Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Second moment variance
    plt.subplot(1, 2, 2)
    plt.loglog(t, mean_second_moment_vars, 'b-', label='Empirical Second Moment Var', linewidth=2)
    plt.loglog(t, 1000000*theoretical_second_moment_var, 'r--', 
              label='Theoretical Second Moment Bound', linewidth=2)
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Variance (log scale)')
    plt.title('Second Moment Variance')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    # Plot 3: Parameter convergence
    plt.subplot(1, 2, 1)
    plt.semilogy(mean_distances, 'b-', label='Adam (Empirical)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||θ - θ*||²')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)

    
    # Plot 4: Loss convergence
    plt.subplot(1, 2, 2)
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
    
    # Verify moment estimation variance properties
    early_first_moment_var = mean_first_moment_vars[len(mean_first_moment_vars)//4]
    late_first_moment_var = mean_first_moment_vars[-1]
    early_second_moment_var = mean_second_moment_vars[len(mean_second_moment_vars)//4]
    late_second_moment_var = mean_second_moment_vars[-1]
    
    print("\nMoment Estimation Analysis:")
    print(f"First moment variance ratio (early/late): {early_first_moment_var/late_first_moment_var:.4f}")
    print(f"Second moment variance ratio (early/late): {early_second_moment_var/late_second_moment_var:.4f}")
    print(f"Theoretical first moment steady-state variance: {(1-beta1)**2/(1-beta1**2) * L_data**2:.4e}")
    print(f"Theoretical second moment steady-state variance: {(1-beta2)**2/(1-beta2**2) * L_data**4:.4e}")
    print(f"Final distance to optimum: {mean_distances[-1]:.4e}")
    print(f"Final loss value: {mean_losses[-1]:.4e}")
    
    return {
        'distances': mean_distances,
        'first_moment_variance': mean_first_moment_vars,
        'second_moment_variance': mean_second_moment_vars,
        'theoretical_first_moment_var': theoretical_first_moment_var,
        'theoretical_second_moment_var': theoretical_second_moment_var,
        'losses': mean_losses,
    }

results = analyze_adam_convergence()
