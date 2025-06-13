"""
Advanced NumPy Toolkit - Practical Task P2

Author: George Dorochov
Email: jordanaftermidnight@gmail.com
Description: Advanced NumPy project with sophisticated algorithms and data processing techniques
"""

__version__ = "1.0.0"
__author__ = "George Dorochov"
__email__ = "jordanaftermidnight@gmail.com"
__description__ = "Advanced NumPy project with sophisticated algorithms and data processing techniques"

# Import main modules
from . import advanced_arrays
from . import signal_processing  
from . import linear_algebra
from . import image_processing
from . import data_analysis

__all__ = [
    'advanced_arrays',
    'signal_processing', 
    'linear_algebra',
    'image_processing',
    'data_analysis'
]