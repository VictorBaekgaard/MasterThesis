# load packages
import torch
import random
import time

random.seed(302)
torch.manual_seed(302)


def SGD(func, lr, n_iter, data, theta_true, tol=1e-5, 
        batch_size=1, theta_init=None, decay_rate=0.1, func_args={}, do_time=False):
    """
    Stochastic Gradient Descent (SGD) implementation with optional real-time tracking.

    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i.
    - lr: Learning rate (η).
    - n_iter: Number of iterations (T).
    - data: Training data, a list of (x_i, y_i) tuples.
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly.
    - decay_rate: A number to gradually reduce step size to improve convergence.
    - func_args: Additional arguments for loss function, dictionary.
    - do_time: Boolean to track real time for convergence visualization.

    Returns:
    - theta: The learned parameter after T iterations.
    - loss_history: A list of loss values for each iteration.
    - theta_history: A list of parameter values for each iteration.
    - time_history: (Optional) A list of real-time timestamps for each iteration (if do_time is True).
    """

    # Initialize theta
    if theta_init is None:
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        theta = theta_init.clone().detach().requires_grad_(True)

    loss_history = []
    theta_history = []
    time_history = []  # For tracking real time if enabled

    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for epoch in range(n_iter):
        # Adjust learning rate
        lr_t = lr / (1 + decay_rate * epoch)

        # Sample a batch
        batch = random.sample(data, batch_size)

        # Initialize gradient
        gradient = torch.zeros_like(theta)

        # Compute gradient for each pair in batch
        for x_i, y_i in batch:
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()
            gradient += theta.grad
            theta.grad.zero_()

        # Mean of the gradient
        gradient /= batch_size

        # Update parameters
        with torch.no_grad():
            theta -= lr_t * gradient

        # Store loss and theta history
        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())

        # If tracking time, append elapsed time for each iteration
        if do_time:
            elapsed_time = time.time() - start_time
            time_history.append(elapsed_time)

        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {epoch}')
            break

        
    # Return time history if do_time is enabled
    if do_time:
        return theta.detach().clone(), loss_history, theta_history, time_history
    else:
        return theta.detach().clone(), loss_history, theta_history

# Mini-Batch SGD
def MiniBatchSGD(func, lr, n_gradients, data, theta_true, tol = 1e-5, 
                 batch_size = 1, decay_rate = 0.1, theta_init=None, func_args = {}, do_time=False):
    """
    Mini-Batch SGD implementation.

    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i.
    - lr: Learning rate (η).
    - n_gradients: Number of gradients calculated.
    - data: Training data, a list of (x_i, y_i) tuples.
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly.
    - devay_rate: A number to graduately reduce step size to improve convergence. 
    - func_arcs: Additional arguments for loss function, dictionary. 
    - do_time: Boolean to track real time for convergence visualization.

    Returns:
    - theta: The learned parameter after T iterations.
    - loss_history: A list of loss values for each iteration.
    - theta_history: A list of parameter values for each iteration. 
    """

    if theta_init is None:
        # Initialize theta randomly based on the dimension of x_i
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        # Use provided initial theta
        theta = theta_init.clone().detach().requires_grad_(True)

    loss_history = []
    theta_history = []
    time_history = []  # For tracking real time if enabled

    n_iter = n_gradients // batch_size
    
    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for epoch in range(n_iter):
        lr_t = lr / (1 + decay_rate * epoch)

        # sample one batch
        batch = random.sample(data, batch_size)

        # to store gradients
        gradient = torch.zeros_like(theta)

        # compute gradient for each pair in batch
        for x_i, y_i in batch:
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()

            gradient += theta.grad
            theta.grad.zero_()

        # mean of the gradient
        gradient /= batch_size

        # update parameter
        with torch.no_grad():
            theta -= lr_t * gradient
        

        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())

        # If tracking time, append elapsed time for each iteration
        if do_time:
            elapsed_time = time.time() - start_time
            time_history.append(elapsed_time)

        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {epoch}')
            break

    # Return time history if do_time is enabled
    if do_time:
        return theta, loss_history, theta_history, time_history
    else:
        return theta, loss_history, theta_history

# SGD with momentum
def MomentumSGD(func, lr, momentum, n_iter, data, theta_true, tol = 1e-5, 
                batch_size = 1, theta_init=None, func_args = {}, do_time=False, track_momentum=False):
    """
    Stochastic Gradient Descent with momentum implementation.

    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i.
    - lr: Learning rate (η).
    - momentum: Momentum parameter 
    - n_iter: Number of iterations (T).
    - data: Training data, a list of (x_i, y_i) tuples.
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly.
    - func_args: Additional arguments for loss function, dictionary. 
    - do_time: Boolean to track real time for convergence visualization.

    Returns:
    - theta: The learned parameter after T iterations.
    - loss_history: A list of loss values for each iteration.
    - theta_history: A list of parameter values for each iteration. 
    - time_history: (Optional) A list of real-time timestamps for each iteration (if do_time is True).
    """

    if theta_init is None:
        # Initialize theta randomly based on the dimension of x_i
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        # Use provided initial theta
        theta = theta_init.clone().detach().requires_grad_(True)

    # Initialize velocity as 0
    velocity = torch.zeros_like(theta)

    loss_history = []
    theta_history = []
    time_history = []  # For tracking real time if enabled
    velocity_history =[] # For tracking the gradient of the velocity
    
    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for epoch in range(n_iter):
        # sample one batch
        batch = random.sample(data, batch_size)

        # to store gradients
        gradient = torch.zeros_like(theta)

        # compute gradient for each pair in batch
        for x_i, y_i in batch:
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()

            gradient += theta.grad
            theta.grad.zero_()

        # mean of the gradient
        gradient /= batch_size

        with torch.no_grad():
            # update velocity
            #m_t = momentum - (1-momentum**(epoch+1))
            #velocity = m_t * velocity + lr * theta.grad 
            #velocity = m_t * velocity + lr * gradient
            velocity = momentum * velocity + lr * gradient
            velocity_norm = torch.norm(velocity)
            # update theta
            theta -= velocity

        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())
        velocity_history.append(velocity_norm)
        # If tracking time, append elapsed time for each iteration
        if do_time:
            elapsed_time = time.time() - start_time
            time_history.append(elapsed_time)

        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {epoch}')
            break

    # Return time history if do_time is enabled
    if do_time:
        return theta, loss_history, theta_history, time_history
    elif track_momentum:
        return theta, loss_history, theta_history, velocity_history
    else:
        return theta, loss_history, theta_history

# Stochastic Variance-Reduced Gradient
def SVRG(func, lr, n_epochs, inner_loop_size, data, theta_true, tol = 1e-5, 
         batch_size = 1, theta_init=None, func_args={}, do_time=False):
    """
    Stochastic Variance-Reduced Gradient (SVRG) implementation.
    
    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i
    - lr: Learning rate 
    - n_epochs: Number of outer loop iterations (s)
    - inner_loop_size: Number of inner loop iterations (t)
    - data: Training data, a list of (x_i, y_i) tuples
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly
    - func_args: Additional arguments for loss function, dictionary
    - do_time: Boolean to track real time for convergence visualization.
    
    Returns:
    - theta: The learned parameter after all iterations
    - loss_history: A list of loss values for each outer iteration
    - theta_history: A list of parameter values for each outer iteration
    - time_history: (Optional) A list of real-time timestamps for each iteration (if do_time is True).
    """
    if theta_init is None:
        # Initialize theta randomly based on the dimension of x_i
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        # Use provided initial theta
        theta = theta_init.clone().detach().requires_grad_(True)
    
    loss_history = []
    theta_history = []
    time_history = []  # For tracking real time if enabled
    n = len(data)
    
    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for s in range(n_epochs):
        # Step 4: Compute full gradient at current point
        full_gradient = torch.zeros_like(theta)
        theta.requires_grad_(True)
        
        for x_i, y_i in data:
            theta.grad = None
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()
            full_gradient += theta.grad
            #theta.grad.zero_()
        
        full_gradient /= n  # Average gradient over all data points
        theta_s = theta.detach().clone()  # Store current parameters
        
        # Inner loop
        for k in range(inner_loop_size):
            # sample one batch
            batch = random.sample(data, batch_size)

            # Compute gradient for the mini-batch
            gradient_current = torch.zeros_like(theta)
            gradient_snapshot = torch.zeros_like(theta)

            for x_i, y_i in batch:
                # Compute gradient at current theta
                theta.requires_grad_(True)
                loss = func(theta, x_i, y_i, **func_args)
                loss.backward()
                gradient_current += theta.grad
                theta.grad.zero_()
                
                # Compute gradient at snapshot theta_s
                theta_s.requires_grad_(True)
                loss_s = func(theta_s, x_i, y_i, **func_args)
                loss_s.backward()
                gradient_snapshot += theta_s.grad
                theta_s.grad.zero_()


            # Average gradients over mini-batch
            gradient_current /= batch_size
            gradient_snapshot /= batch_size
            
            # Compute variance-reduced gradient
            v_grad = gradient_current - gradient_snapshot + full_gradient
            
            # Update parameters
            with torch.no_grad():
                theta = theta - lr * v_grad

            # If tracking time, append elapsed time for each iteration
            if do_time:
                elapsed_time = time.time() - start_time
                time_history.append(elapsed_time)    

            
        # Store history
        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())
    
        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {s}')
            break

    # Return time history if do_time is enabled
    if do_time:
        return theta, loss_history, theta_history, time_history
    else:
        return theta, loss_history, theta_history

# AdaGrad
def AdaGrad(func, lr, n_iter, data, theta_true, tol = 1e-5, 
            precision = 10e-8, batch_size = 1, theta_init=None, func_args = {}, do_time=False):
    """
    Adaptive Gradient Algorithm (AdaGrad) implementation.

    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i.
    - lr: Initial Learning rate (η).
    - n_iter: Number of iterations (T).
    - data: Training data, a list of (x_i, y_i) tuples.
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - precision: Small value for numerical stability.
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly.
    - func_args: Additional arguments for loss function, dictionary. 
    - do_time: Boolean to track real time for convergence visualization.

    Returns:
    - theta: The learned parameter after T iterations.
    - loss_history: A list of loss values for each iteration.
    - theta_history: A list of parameter values for each iteration. 
    - time_history: (Optional) A list of real-time timestamps for each iteration (if do_time is True).
    """

    if theta_init is None:
        # Initialize theta randomly based on the dimension of x_i
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        # Use provided initial theta
        theta = theta_init.clone().detach().requires_grad_(True)

    # Initialize acc. squared gradient as 0
    G = torch.zeros_like(theta)

    loss_history = []
    theta_history = []
    time_history = []  # For tracking real time if enabled
    
    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for epoch in range(n_iter):
        # sample one batch
        batch = random.sample(data, batch_size)

        # to store gradients
        gradient = torch.zeros_like(theta)

        # compute gradient for each pair in batch
        for x_i, y_i in batch:
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()

            gradient += theta.grad
            theta.grad.zero_()

        # mean of the gradient
        gradient /= batch_size
        
        with torch.no_grad():
            # update G_t
            G += gradient ** 2
            # update learning rate
            lr_t = lr / (torch.sqrt(G)+precision) 
            # update theta
            theta -= lr_t * gradient

        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())

        # If tracking time, append elapsed time for each iteration
        if do_time:
            elapsed_time = time.time() - start_time
            time_history.append(elapsed_time)

        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {epoch}')
            break

    # Return time history if do_time is enabled
    if do_time:
        return theta, loss_history, theta_history, time_history
    else:
        return theta, loss_history, theta_history

# Adam
def Adam(func, lr, beta_1, beta_2, n_iter, data, theta_true, tol = 1e-5, 
         precision = 10e-8, batch_size = 1, theta_init=None, func_args = {}, do_time=False, return_moments=False):
    """
    Adam implementation.

    Parameters:
    - func: A function that computes the loss given theta, x_i, y_i.
    - lr: Learning rate (η).
    - beta_1: Decay parameter 1.
    - beta_2: Decay parameter 2.
    - n_iter: Number of iterations (T).
    - data: Training data, a list of (x_i, y_i) tuples.
    - theta_true: True parameter of the objective
    - tol: Tolerance value for early stopping. Used in norm-distance from true parameter value.
    - precision: Small number for numerical stability. 
    - batch_size: Number of data points for each batch
    - theta_init: Initial parameter tensor. If None, initialized randomly.
    - func_args: Additional arguments for loss function, dictionary. 
    - do_time: Boolean to track real time for convergence visualization.

    Returns:
    - theta: The learned parameter after T iterations.
    - loss_history: A list of loss values for each iteration.
    - theta_history: A list of parameter values for each iteration. 
    - time_history: (Optional) A list of real-time timestamps for each iteration (if do_time is True).
    """

    if theta_init is None:
        # Initialize theta randomly based on the dimension of x_i
        input_dim = data[0][0].shape
        theta = torch.randn(*input_dim, requires_grad=True)
    else:
        # Use provided initial theta
        theta = theta_init.clone().detach().requires_grad_(True)

    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)

    loss_history = []
    theta_history = []
    moment_history = []
    time_history = []  # For tracking real time if enabled
    
    # Start timing if do_time is True
    if do_time:
        start_time = time.time()

    for epoch in range(n_iter):
        
        # sample one batch
        batch = random.sample(data, batch_size)

        # to store gradients
        gradient = torch.zeros_like(theta)

        # compute gradient for each pair in batch
        for x_i, y_i in batch:
            loss = func(theta, x_i, y_i, **func_args)
            loss.backward()

            gradient += theta.grad
            theta.grad.zero_()

        # mean of the gradient
        gradient /= batch_size
        
        # Zero the gradients
        if theta.grad is not None:
            theta.grad.zero_()
        
        
        # Compute the loss for the data point
        loss = func(theta, x_i, y_i, **func_args)
        loss.backward()

        
        with torch.no_grad():
            # update first moment
            m = beta_1 * m + (1-beta_1)*gradient

            # update second moment
            v = beta_2 * v + (1-beta_2)*gradient**2

            # bias correction
            m_hat = m/(1-beta_1**(epoch +1))
            v_hat = v/(1-beta_2**(epoch +1))
            
            # update theta
            theta -= lr * m_hat/(torch.sqrt(v_hat) + precision)

        loss_history.append(loss.item())
        theta_history.append(theta.detach().clone())
        if return_moments:
            moment_history.append((m.clone(), v.clone()))

        # If tracking time, append elapsed time for each iteration
        if do_time:
            elapsed_time = time.time() - start_time
            time_history.append(elapsed_time)

        # Early stopping based on convergence to theta_true
        distance = torch.norm(theta - theta_true).item()
        if distance < tol:
            print(f'Converged to theta_true within tolerance at iteration {epoch}')
            break
        
    # Return time history if do_time is enabled
    if do_time:
        return theta, loss_history, theta_history, time_history
    elif return_moments: 
        return theta, loss_history, theta_history, moment_history
    else:
        return theta, loss_history, theta_history
