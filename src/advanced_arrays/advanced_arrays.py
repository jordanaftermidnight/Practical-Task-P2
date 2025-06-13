"""Advanced Array Operations with NumPy"""
import numpy as np
from typing import Tuple, List, Optional, Union
import warnings

class AdvancedArrays:
    """
    Advanced array operations showcasing sophisticated NumPy techniques.
    """
    
    def __init__(self):
        self.random_state = 42
        np.random.seed(self.random_state)
    
    def fancy_indexing_demo(self, shape: Tuple[int, ...] = (100, 100)) -> np.ndarray:
        """
        Demonstrate advanced indexing techniques including boolean indexing,
        fancy indexing, and multi-dimensional slicing.
        """
        # Create sample data
        data = np.random.randn(*shape)
        
        # Boolean indexing: select elements based on conditions
        mask = (data > 0.5) & (data < 2.0)
        selected_values = data[mask]
        
        # Fancy indexing: use arrays as indices
        rows = np.array([10, 20, 30, 40, 50])
        cols = np.array([15, 25, 35, 45, 55])
        fancy_indexed = data[rows[:, np.newaxis], cols]
        
        # Advanced slicing with step
        sliced_data = data[::2, ::3]
        
        return {
            'original_shape': data.shape,
            'boolean_selected': selected_values,
            'fancy_indexed': fancy_indexed,
            'sliced_shape': sliced_data.shape,
            'original_data': data
        }
    
    def advanced_broadcasting(self, a_shape: Tuple[int, ...] = (5, 1, 3), 
                            b_shape: Tuple[int, ...] = (1, 4, 1)) -> dict:
        """
        Demonstrate advanced broadcasting operations.
        """
        a = np.random.randn(*a_shape)
        b = np.random.randn(*b_shape)
        
        # Broadcasting multiplication
        broadcasted_mult = a * b
        
        # Broadcasting with custom operations
        broadcasted_custom = np.where(a > 0, a * b, a / (b + 1e-8))
        
        # Outer product using broadcasting
        x = np.random.randn(10)
        y = np.random.randn(15)
        outer_product = x[:, np.newaxis] * y[np.newaxis, :]
        
        return {
            'a_shape': a.shape,
            'b_shape': b.shape,
            'broadcasted_shape': broadcasted_mult.shape,
            'broadcasted_mult': broadcasted_mult,
            'broadcasted_custom': broadcasted_custom,
            'outer_product': outer_product
        }
    
    def structured_arrays_demo(self) -> np.ndarray:
        """
        Demonstrate structured arrays (record arrays) for complex data.
        """
        # Define a structured array dtype
        dtype = np.dtype([
            ('name', 'U20'),
            ('age', 'i4'),
            ('height', 'f4'),
            ('weight', 'f4'),
            ('scores', '3f4')  # Array of 3 floats
        ])
        
        # Create structured array
        data = np.array([
            ('Alice', 25, 165.5, 55.2, [85.5, 92.1, 78.3]),
            ('Bob', 30, 180.0, 75.8, [90.0, 88.5, 95.2]),
            ('Charlie', 28, 175.2, 68.5, [82.1, 85.7, 89.9]),
            ('Diana', 26, 160.8, 52.3, [94.5, 91.2, 87.8]),
            ('Eve', 32, 170.1, 62.1, [88.3, 93.6, 85.4])
        ], dtype=dtype)
        
        # Advanced operations on structured arrays
        # Sort by age
        sorted_by_age = np.sort(data, order='age')
        
        # Filter by condition
        tall_people = data[data['height'] > 170]
        
        # Compute BMI
        bmi = data['weight'] / (data['height'] / 100) ** 2
        
        # Average scores
        avg_scores = np.mean(data['scores'], axis=1)
        
        return {
            'original_data': data,
            'sorted_by_age': sorted_by_age,
            'tall_people': tall_people,
            'bmi': bmi,
            'average_scores': avg_scores
        }
    
    def memory_views_and_strides(self, shape: Tuple[int, ...] = (1000, 1000)) -> dict:
        """
        Demonstrate memory views and stride manipulation for efficient operations.
        """
        # Create large array
        data = np.random.randn(*shape)
        
        # Create different views without copying data
        transposed_view = data.T
        reshaped_view = data.reshape(-1, 10)
        
        # Stride manipulation for efficient access patterns
        # Create a sliding window view
        window_size = 5
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            data[0, :100], window_shape=window_size
        )
        
        # Memory-efficient operations using views
        # Compute statistics along different axes
        row_means = np.mean(data, axis=1)
        col_stds = np.std(data, axis=0)
        
        return {
            'original_shape': data.shape,
            'transposed_shape': transposed_view.shape,
            'reshaped_shape': reshaped_view.shape,
            'windowed_shape': windowed_data.shape,
            'memory_shared': np.shares_memory(data, transposed_view),
            'row_means_shape': row_means.shape,
            'col_stds_shape': col_stds.shape,
            'original_strides': data.strides,
            'transposed_strides': transposed_view.strides
        }
    
    def advanced_ufuncs(self) -> dict:
        """
        Demonstrate advanced universal function operations.
        """
        # Create sample data
        x = np.linspace(-5, 5, 1000)
        y = np.linspace(-3, 3, 1000)
        X, Y = np.meshgrid(x, y)
        
        # Custom ufunc using np.frompyfunc
        def custom_sigmoid(x, alpha=1.0):
            """Custom sigmoid function"""
            return 1 / (1 + np.exp(-alpha * x))
        
        # Create ufunc
        sigmoid_ufunc = np.frompyfunc(custom_sigmoid, 2, 1)
        
        # Apply ufunc
        Z = sigmoid_ufunc(X + Y, 2.0).astype(float)
        
        # Advanced reduction operations
        # Custom reduction using np.ufunc.reduce
        data = np.random.randn(10, 10)
        
        # Cumulative operations
        cumsum_result = np.cumsum(data, axis=0)
        cumprod_result = np.cumprod(np.abs(data), axis=1)
        
        # Broadcasting with ufuncs
        angles = np.linspace(0, 2*np.pi, 100)
        sin_cos_combined = np.sin(angles[:, np.newaxis]) * np.cos(angles[np.newaxis, :])
        
        return {
            'meshgrid_shapes': (X.shape, Y.shape),
            'custom_ufunc_result': Z,
            'cumsum_shape': cumsum_result.shape,
            'cumprod_shape': cumprod_result.shape,
            'sin_cos_combined_shape': sin_cos_combined.shape,
            'sample_values': {
                'Z_sample': Z[::100, ::100],
                'cumsum_sample': cumsum_result[:5, :5],
                'sin_cos_sample': sin_cos_combined[:5, :5]
            }
        }
    
    def vectorized_operations_demo(self) -> dict:
        """
        Demonstrate vectorized operations for performance optimization.
        """
        # Large dataset
        n_samples = 100000
        data = np.random.randn(n_samples, 10)
        
        # Vectorized statistical operations
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        
        # Vectorized distance calculations
        # Euclidean distances between all pairs of first 1000 samples
        subset = data[:1000]
        distances = np.sqrt(np.sum((subset[:, np.newaxis] - subset[np.newaxis, :]) ** 2, axis=2))
        
        # Vectorized conditional operations
        normalized_data = np.where(stds > 0, (data - means) / stds, data)
        
        # Vectorized complex operations
        # Compute correlation matrix efficiently
        correlation_matrix = np.corrcoef(data.T)
        
        # Vectorized boolean operations
        outlier_mask = np.abs(normalized_data) > 3
        outlier_counts = np.sum(outlier_mask, axis=0)
        
        return {
            'data_shape': data.shape,
            'means': means,
            'stds': stds,
            'distance_matrix_shape': distances.shape,
            'correlation_matrix_shape': correlation_matrix.shape,
            'outlier_counts': outlier_counts,
            'normalized_stats': {
                'mean': np.mean(normalized_data, axis=0),
                'std': np.std(normalized_data, axis=0)
            }
        }