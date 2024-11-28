# load packages
import torch

# Loss functions for each scenario
def quadratic_loss(theta, x_i, y_i, A, b):
    return 0.5 * (theta @ A @ theta) + b @ theta

def absolute_loss(theta, x_i, y_i):
    pred = torch.dot(x_i, theta)
    return torch.abs(pred - y_i)

def nonconvex_loss(theta, x_i, y_i):
    pred = torch.sin(2 * torch.dot(x_i, theta))
    return (pred - y_i)**2

def ridge_regression_loss(theta, x_i, y_i, lambda_reg):
    """
    Compute the Ridge Regression loss for a single sample.

    Parameters:
    - theta: Parameter vector of shape (n_features,).
    - x_i: Feature vector of shape (n_features,).
    - y_i: Target scalar.
    - lambda_reg: Regularization parameter.

    Returns:
    - loss: Scalar tensor.
    """
    # Ensure theta requires gradient
    theta.requires_grad_(True)

    # Compute prediction
    y_pred = torch.dot(theta, x_i)

    # Compute loss
    loss = 0.5 * (y_pred - y_i) ** 2 + 0.5 * lambda_reg * torch.sum(theta ** 2)
    return loss