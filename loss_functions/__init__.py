# loss_functions/__init__.py
from .loss_functions import quadratic_loss, absolute_loss, nonconvex_loss, ridge_regression_loss  # Import functions from loss_functions.py

__all__ = ['quadratic_loss', 
           'absolute_loss', 
           'nonconvex_loss',
           'ridge_regression_loss']
