# generate_data/__init__.py
from .data_generator import generate_quadratic_data, generate_convex_data, generate_nonconvex_data, generate_strongly_convex_data
__all__ = ['generate_quadratic_data', 
           'generate_convex_data', 
           'generate_nonconvex_data',
           'generate_strongly_convex_data']  # Defines the public API for this package
