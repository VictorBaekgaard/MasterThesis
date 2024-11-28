import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import numpy as np
import algorithms, loss_functions, generate_data, visualize

plt.rcParams['figure.dpi'] = 150

def analyze_linear_convergence(n_samples: int = 1000, d: int = 50, n_runs: int = 1):
    """
    Analyze linear convergence rate for SVRG on strongly convex problems
    with careful parameter tuning
    """
    # Problem parameters
    lambda_reg = 0.1  # Strong convexity parameter
    func_args = {'lambda_reg': lambda_reg}
    
    # Generate problem instance to compute Lipschitz constant
    X, y, theta_true = generate_data.generate_strongly_convex_data(
        n_samples=n_samples, n_features=d, noise_std=0.01, random_state=302
    )
    data = list(zip(X, y))
    
    # Estimate Lipschitz constant L (this is problem dependent)
    # For ridge regression, L ≈ largest eigenvalue of X^T X / n + lambda
    X_tensor = torch.stack([x for x, _ in data])
    L = torch.linalg.norm(X_tensor.T @ X_tensor / n_samples) + lambda_reg
    
    # SVRG parameters
    # Set learning rate based on theory: need η ≤ 1/(10L) for linear convergence
    lr = 1.0 / (10 * L)
    n_epochs = 30  # Increased to see convergence better
    inner_loop_size = n_samples // 4 # Set to dataset size as per theory
    
    # Storage for runs
    svrg_distances_all = []
    sgd_distances_all = []

    svrg_loss_all = []
    sgd_loss_all = []
    
    for run in range(n_runs):
        print(f"Run number {run + 1} \n")
        # Generate new problem instance
        X, y, theta_true = generate_data.generate_strongly_convex_data(
            n_samples=n_samples, n_features=d, noise_std=0.1, random_state=302+run
        )
        data = list(zip(X, y))
        theta_init = torch.randn_like(theta_true)  # Random initialization
        
        # Run SVRG
        print("Running svrg... \n")
        svrg_results = algorithms.SVRG(
            func=loss_functions.ridge_regression_loss,
            lr=lr,
            n_epochs=n_epochs,
            inner_loop_size=inner_loop_size,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args
        )
        
        # Run SGD for comparison
        print("Running SGD... \n")
        sgd_results = algorithms.SGD(
            func=loss_functions.ridge_regression_loss,
            lr=lr,  # Same learning rate for fair comparison
            n_iter=n_epochs * inner_loop_size,  # Match total iterations
            data=data,
            theta_true=theta_true,
            batch_size=1,
            decay_rate=0.0,
            theta_init=theta_init.clone(),
            func_args=func_args
        )
        
        # Calculate distances - use optimization error ||θ_t - θ*||²
        svrg_distances = [torch.norm(theta - theta_true).item()**2 
                         for theta in svrg_results[2]]
        sgd_distances = [torch.norm(theta - theta_true).item()**2 
                        for theta in sgd_results[2]]
        
        svrg_loss_all.append(svrg_results[1])
        sgd_loss_all.append(sgd_results[1])

        svrg_distances_all.append(svrg_distances)
        sgd_distances_all.append(sgd_distances)
    


    # Convert to numpy arrays
    svrg_distances_all = np.array(svrg_distances_all)
    sgd_distances_all = np.array(sgd_distances_all)

    # Calculate means
    svrg_mean = np.mean(svrg_distances_all, axis=0)
    sgd_mean = np.mean(sgd_distances_all, axis=0)
    #svrg_mean = np.repeat(svrg_mean, inner_loop_size)
    sgd_mean = sgd_mean[::inner_loop_size]
    svrg_loss_mean = np.mean(np.array(svrg_loss_all), axis=0)
    sgd_loss_mean = np.mean(np.array(sgd_loss_all), axis=0)


    # Theoretical rate computation
    # For SVRG: ρ = (1 - μ/(10L)) where μ is strong convexity parameter
    mu = lambda_reg  # Strong convexity parameter
    theoretical_rate = 1 - mu/(10*L)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot empirical convergence
    iterations = np.arange(1, len(sgd_mean) + 1)
    
    # Plot in log scale
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, svrg_mean, 'g-', label='SVRG (empirical)', linewidth=2)
    plt.semilogy(iterations, sgd_mean, 'b-', label='SGD (empirical)', linewidth=2)
    
    # Plot theoretical rate
    initial_error = svrg_mean[0]
    theoretical_convergence = initial_error * np.power(theoretical_rate, iterations)
    plt.semilogy(iterations, theoretical_convergence, 'r--', 
                label='SVRG (theoretical)', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Optimization Error (log scale)')
    plt.title('Convergence in Log Scale')
    plt.legend()
    plt.grid(True)
    
    # Also plot ratio of consecutive errors to verify linear rate
    plt.subplot(1, 2, 2)
    svrg_ratios = svrg_mean[1:] / svrg_mean[:-1]
    sgd_ratios = sgd_mean[1:] / sgd_mean[:-1]
    
    plt.plot(iterations[1:], svrg_ratios, 'g-', label='SVRG ratio', linewidth=2)
    plt.plot(iterations[1:], sgd_ratios, 'b-', label='SGD ratio', linewidth=2)
    plt.axhline(y=theoretical_rate, color='r', linestyle='--',
                label='Theoretical rate', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Error Ratio (θ_{t+1}/θ_t)')
    plt.title('Convergence Rate Verification')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute empirical rates
    def compute_convergence_rate(distances):
        # Use latter half of iterations to compute rate (after initial phase)
        start_idx = len(distances) // 2
        ratios = distances[start_idx+1:] / distances[start_idx:-1]
        return np.mean(ratios)
    
    svrg_rate = compute_convergence_rate(svrg_mean)
    sgd_rate = compute_convergence_rate(sgd_mean)
    
    print("\nConvergence Rate Analysis:")
    print(f"Problem Parameters:")
    print(f"L (Lipschitz constant): {L:.4f}")
    print(f"μ (strong convexity): {mu:.4f}")
    print(f"Learning rate: {lr:.4e}")
    print(f"\nEmpirical convergence rates:")
    print(f"SVRG: {svrg_rate:.4f}")
    print(f"SGD: {sgd_rate:.4f}")
    print(f"Theoretical SVRG rate: {theoretical_rate:.4f}")
    print(svrg_loss_mean)
    print(sgd_loss_mean)
    return {
        'svrg_rate': svrg_rate,
        'sgd_rate': sgd_rate,
        'theoretical_rate': theoretical_rate,
        'lipschitz': L.item(),
        'learning_rate': lr
    }

def analyze_linear_convergence1(n_samples: int = 1000, d: int = 10, n_runs: int = 1):
    """
    Analyze linear convergence rate for SVRG accounting for inner loop iterations
    """
    # Problem parameters
    lambda_reg = 0.01
    func_args = {'lambda_reg': lambda_reg}

    # Generate problem instance
    X, y, theta_true = generate_data.generate_strongly_convex_data(
        n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302
    )

    data = list(zip(X, y))
    
    # Estimate Lipschitz constant
    X_tensor = torch.stack([x for x, _ in data])
    L = torch.linalg.norm(X_tensor.T @ X_tensor / n_samples) + lambda_reg
    
    eigenvalues = torch.linalg.eigvalsh(X_tensor.T @ X_tensor / n_samples)
    L_data = eigenvalues.max().item() + lambda_reg
    mu_data = eigenvalues.min().item() + lambda_reg
    condition_number = L_data / mu_data
    print(f"Data condition number: {condition_number:.2f}")


    L = L_data
    # SVRG parameters
    lr = 1.0 / (10*L)
    n_epochs = 20
    inner_loop_size = int(5*(L/mu_data)) #n_samples // 4
    total_iterations = n_epochs * inner_loop_size
    
    print(f"\nSVRG Configuration:")
    print(f"Outer loops (epochs): {n_epochs}")
    print(f"Inner loop size: {inner_loop_size}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    svrg_distances_all = []
    
    for run in range(n_runs):
        theta_init = torch.zeros_like(theta_true) 

        svrg_results = algorithms.SVRG(
            func=loss_functions.ridge_regression_loss,
            lr=lr,
            n_epochs=n_epochs,
            inner_loop_size=inner_loop_size,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args
        )
        
        # Get distances from each outer loop
        svrg_distances = [torch.norm(theta - theta_true).item()**2 
                         for theta in svrg_results[2]]
        svrg_distances_all.append(svrg_distances)
    
    # Convert to numpy array and compute mean
    svrg_distances_all = np.array(svrg_distances_all)
    svrg_mean = np.mean(svrg_distances_all, axis=0)
    
    # Compute theoretical convergence rate
    mu = lambda_reg
    mu = mu_data
    theoretical_rate = 1 - mu/(16*L)
    
    # Create theoretical line considering inner iterations
    # Interpolate between outer loop measurements
    # Compute theoretical error over epochs

    # Compute theoretical convergence rate per epoch
    gamma = 16  # This constant may vary based on theoretical analysis
    theoretical_epoch_rate = 1 - mu / (gamma * L)

    outer_iterations = np.arange(len(svrg_mean))
    theoretical_outer = svrg_mean[0] * np.power(theoretical_epoch_rate, outer_iterations*inner_loop_size)

    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Outer loop measurements with theoretical line
    plt.subplot(1, 2, 1)
    plt.semilogy(outer_iterations, svrg_mean, 'go-', label='SVRG (Empirical)', linewidth=2)
    plt.semilogy(outer_iterations, theoretical_outer, 'r--', 
                label='SVRG (Theoretical)', linewidth=2)
    plt.xlabel('Epoch (Outer Loop)')
    plt.ylabel('||θ - θ*||²')
    plt.title('Convergence: Outer Loop View')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Theoretical inner loop convergence
    inner_iterations = np.arange(total_iterations)
    theoretical_inner = svrg_mean[0] * np.power(theoretical_rate, inner_iterations)
    
    # Plot 3: Convergence rates
    plt.subplot(1, 2, 2)
    # Compute rates between outer loop measurements
    outer_rates = svrg_mean[1:] / svrg_mean[:-1]
    # Compute expected rate over inner_loop_size iterations
    expected_bulk_rate = theoretical_rate ** inner_loop_size
    
    plt.plot(outer_rates, 'go-', label='Empirical (per epoch)', linewidth=2)
    plt.axhline(y=expected_bulk_rate, color='r', linestyle='--',
                label=f'Expected rate per epoch\n({theoretical_rate:.4f}^{inner_loop_size})')
    plt.xlabel('Epoch')
    plt.ylabel('Error Ratio')
    plt.title('Convergence Rates')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nConvergence Analysis:")
    print(f"L (Lipschitz constant): {L:.4f}")
    print(f"μ (strong convexity): {mu:.4f}")
    print(f"Learning rate: {lr:.4e}")
    print(f"Theoretical rate per inner iteration: {theoretical_rate:.4f}")
    print(f"Expected rate per epoch: {expected_bulk_rate:.4f}")
    print("\nError reduction:")
    print(f"After 1 epoch (theoretical): {(theoretical_rate**inner_loop_size):.4f}")
    print(f"After 1 epoch (empirical): {(svrg_mean[1]/svrg_mean[0]):.4f}")
    
    return {
        'empirical_distances': svrg_mean,
        'theoretical_inner': theoretical_inner,
        'params': {
            'L': L,
            'mu': mu,
            'rate_per_iter': theoretical_rate,
            'rate_per_epoch': expected_bulk_rate
        }
    }

def analyze_sublinear_convergence(n_samples: int = 1000, d: int = 5, n_runs: int = 5):
    """
    Analyze sublinear convergence rate for SVRG in convex (non-strongly convex) settings
    """
    # Generate problem instance for convex data
    data, theta_true = generate_data.generate_convex_data(N=n_samples, d=d)
    
    # Estimate Lipschitz constant
    X_tensor = torch.stack([x for x, _ in data])
    L = torch.linalg.norm(X_tensor.T @ X_tensor / n_samples)
    
    # SVRG parameters
    lr = 1.0 / (100*L)
    n_epochs = 100
    inner_loop_size = n_samples // 4
    total_iterations = n_epochs * inner_loop_size
    
    print(f"\nSVRG Configuration for Convex Case:")
    print(f"Outer loops (epochs): {n_epochs}")
    print(f"Inner loop size: {inner_loop_size}")
    print(f"Total iterations: {total_iterations}")
    
    # Storage for runs
    svrg_distances_all = []
    
    for run in range(n_runs):
        theta_init = torch.zeros_like(theta_true)
        data, theta_true = generate_data.generate_convex_data(N=n_samples, d=d, random_state=302+run)
        # Modify SVRG to track every inner iteration
        svrg_results = algorithms.SVRG(
            func=loss_functions.absolute_loss,
            lr=lr,
            n_epochs=n_epochs,
            inner_loop_size=inner_loop_size,
            data=data,
            theta_true=theta_true,
            theta_init=theta_init.clone()
        )
        
        # Get distances from each outer loop
        svrg_distances = [torch.norm(theta - theta_true).item()**2 
                          for theta in svrg_results[2]]
        svrg_distances_all.append(svrg_distances)
    
    # Convert to numpy array and compute mean
    svrg_distances_all = np.array(svrg_distances_all)
    svrg_mean = np.mean(svrg_distances_all, axis=0)
    
    # Theoretical convergence rate (O(1/k) rate)
    outer_iterations = np.arange(1, len(svrg_mean) + 1)
    theoretical_outer = svrg_mean[0] / outer_iterations  # O(1/k)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Outer loop measurements with theoretical line
    #plt.subplot(1, 2, 1)
    plt.semilogy(outer_iterations, svrg_mean, 'go-', label='SVRG (Empirical)', linewidth=2)
    plt.semilogy(outer_iterations, theoretical_outer, 'r--', 
                 label='SVRG (Theoretical O(1/k))', linewidth=2)
    plt.xlabel('Epoch (Outer Loop)')
    plt.ylabel('||θ - θ*||²')
    plt.title('Convergence: Outer Loop View')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Convergence rates
    '''plt.subplot(1, 2, 2)
    outer_rates = svrg_mean[1:] / svrg_mean[:-1]
    plt.plot(outer_rates, 'go-', label='Empirical Rate per Epoch', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Error Ratio')
    plt.title('Convergence Rates (Empirical)')
    plt.legend()
    plt.grid(True)
    '''
    plt.tight_layout()
    plt.show()
    
    print("\nConvergence Analysis (Convex Case):")
    print(f"L (Lipschitz constant): {L:.4f}")
    print(f"Learning rate: {lr:.4e}")
    
    return {
        'empirical_distances': svrg_mean,
        'theoretical_outer': theoretical_outer,
        'params': {
            'L': L,
            'rate': 'O(1/k)'
        }
    }

def analyse_inner_loop(n_samples=1000, d=5, n_runs=1):
    # Problem parameters
    lambda_reg = 0.1
    func_args = {'lambda_reg': lambda_reg}

    # Generate problem instance
    X, y, theta_true = generate_data.generate_strongly_convex_data(
        n_samples=n_samples, n_features=d, noise_std=0.01, condition_number=15.0, random_state=302
    )

    data = list(zip(X, y))

    # Estimate Lipschitz constant
    X_tensor = torch.stack([x for x, _ in data])
    L = torch.linalg.norm(X_tensor.T @ X_tensor / n_samples) + lambda_reg
    
    eigenvalues = torch.linalg.eigvalsh(X_tensor.T @ X_tensor / n_samples)
    L_data = eigenvalues.max().item() + lambda_reg
    mu_data = eigenvalues.min().item() + lambda_reg
    condition_number = L_data / mu_data
    print(f"Data condition number: {condition_number:.2f}")

    L = L_data
    # SVRG parameters
    lr = 1.0 / (10 * L)
    n_epochs = 10
    inner_loops = [250, 500, 1000]
    
    theta_init = torch.zeros_like(theta_true)
    averaged_results = []

    for loop in inner_loops:
        # Initialize list to store the results from multiple runs
        loop_results = []
        
        for _ in range(n_runs):
            svrg_results = algorithms.SVRG(
                func=loss_functions.ridge_regression_loss,
                lr=lr,
                n_epochs=n_epochs,
                inner_loop_size=loop,
                data=data,
                theta_true=theta_true,
                theta_init=theta_init.clone(),
                func_args=func_args
            )
            # Collect the loss trajectory from this run
            loop_results.append(svrg_results[1])
        
        # Calculate the mean loss trajectory across n_runs
        avg_result = np.mean(loop_results, axis=0)
        averaged_results.append(avg_result)

    # Plot the mean loss values for each inner loop
    for i in range(len(averaged_results)):
        plt.semilogy(averaged_results[i], 'o-', label=f"m = {inner_loops[i]}")
    
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Impact of Inner Loop Length (Averaged over Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return averaged_results


def run_all_experiments():
    """
    Run comprehensive experiments focusing on linear convergence analysis
    """
    #print("Analyzing Linear Convergence...")
    #linear_results = analyze_linear_convergence()
    
    #print("Analysign inner loops")
    #analyse_inner_loop()

    # Only run nonconvex analysis if you have implemented the necessary functions
    # print("Analyzing Nonconvex Behavior...")
    # nonconvex_results = analyze_nonconvex_behavior()


if __name__ == "__main__":
    results1 = analyze_linear_convergence1()
    results2 = analyze_sublinear_convergence()
    results2 = analyse_inner_loop(n_runs=3)