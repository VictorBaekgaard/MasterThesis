# load packages
import numpy as np
import torch

def generate_quadratic_data(N, d=2, condition_number=1.0):
    """
    Generates data for strongly convex quadratic optimization
    where the minimum of the loss function is guaranteed to be at theta_true.
    """
    torch.manual_seed(302)
    print(" --- generating strongly convex data ---\n")
    
    # Create positive definite matrix A with specified condition number
    U, _ = torch.linalg.qr(torch.randn(d, d))
    eigenvalues = torch.logspace(0, np.log10(condition_number), d)
    A = U @ torch.diag(eigenvalues) @ U.T
    
    # Generate true theta
    theta_true = torch.randn(d)*3
    
    # Compute b 
    b = -A @ theta_true
    
    # Generate data points (x_i, y_i)
    data = []
    for _ in range(N):
        x_i = torch.randn(d)
        y_i = x_i @ theta_true
        data.append((x_i, y_i))

    print(" ---------------- DONE ----------------- ")
    return data, theta_true, A, b

def generate_convex_data(N, d=2, random_state = 302):
    """
    Generates data for convex but not strongly convex optimization.
    Uses absolute value loss: f(θ) = |xᵀθ - y|
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    print(" --- generating convex data --- \n")
    
    theta_true = torch.randn(d)
    data = []
    for _ in range(N):
        x_i = torch.randn(d)
        y_i = torch.dot(x_i, theta_true) + 0.1 * torch.randn(1)
        data.append((x_i, y_i))

    print(" ------------ DONE ------------ ")
    return data, theta_true

def generate_nonconvex_data(N, d=2):
    """
    Generates data for non-convex optimization.
    Uses a simple neural network loss with sine activation.
    """
    torch.manual_seed(302)
    print(" --- generating non-convex data --- \n")
    
    theta_true = torch.randn(d)
    data = []
    for _ in range(N):
        x_i = torch.randn(d)
        # Non-convex function using sine
        y_i = torch.sin(2 * torch.dot(x_i, theta_true))
        data.append((x_i, y_i))

    print(" -------------- DONE -------------- ")
    return data, theta_true

import torch

def generate_strongly_convex_data(n_samples=1000, n_features=50, noise_std=0.1, 
                                condition_number=10.0, min_eigenval=1.0,
                                random_state=None):
    """
    Generate data for strongly convex optimization with controlled condition number.
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Number of features
    - noise_std: Standard deviation of the Gaussian noise
    - condition_number: Desired condition number of the covariance matrix
    - min_eigenval: Minimum eigenvalue to ensure strong convexity
    - random_state: Random seed for reproducibility
    
    Returns:
    - X: Feature matrix of shape (n_samples, n_features)
    - y: Target vector of shape (n_samples,)
    - beta_true: True parameter vector of shape (n_features,)
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    
    # Generate true parameters
    beta_true = torch.randn(n_features)
    
    # Step 1: Generate random orthogonal matrix Q using QR decomposition
    A = torch.randn(n_features, n_features)
    Q, R = torch.linalg.qr(A)
    
    # Step 2: Create desired eigenvalue spectrum
    max_eigenval = min_eigenval * condition_number
    eigenvals = torch.exp(torch.linspace(torch.log(torch.tensor(min_eigenval)), 
                                       torch.log(torch.tensor(max_eigenval)), 
                                       n_features))
    
    # Step 3: Construct positive definite covariance matrix
    # Σ = QΛQ^T where Λ is diagonal matrix of eigenvalues
    Sigma = Q @ (eigenvals.diag()) @ Q.T
    
    # Ensure symmetry (numerical stability)
    Sigma = (Sigma + Sigma.T) / 2
    
    # Step 4: Generate features using multivariate normal distribution
    mean = torch.zeros(n_features)
    mvn = torch.distributions.MultivariateNormal(mean, Sigma)
    X = mvn.sample((n_samples,))
    
    # Generate targets with noise
    epsilon = torch.randn(n_samples) * noise_std
    y = X @ beta_true + epsilon


    # Verify positive definiteness
    eigenvals_check = torch.linalg.eigvalsh(Sigma)
    assert torch.all(eigenvals_check > 0), "Covariance matrix is not positive definite"
    actual_condition = eigenvals_check[-1] / eigenvals_check[0]
    
    #print(f"Actual condition number: {actual_condition:.2f}")
    #print(f"Minimum eigenvalue: {eigenvals_check[0]:.6f}")
    
    return X, y, beta_true
