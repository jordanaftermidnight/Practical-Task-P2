"""
Test suite for advanced array operations.
"""
import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_arrays import AdvancedArrays

class TestAdvancedArrays:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.arrays = AdvancedArrays()
    
    def test_fancy_indexing_demo(self):
        """Test fancy indexing operations."""
        result = self.arrays.fancy_indexing_demo((10, 10))
        
        assert 'original_shape' in result
        assert result['original_shape'] == (10, 10)
        assert 'boolean_selected' in result
        assert 'fancy_indexed' in result
        assert 'sliced_shape' in result
        assert 'original_data' in result
        
        # Check that boolean indexing returns a 1D array
        assert result['boolean_selected'].ndim == 1
        
        # Check fancy indexing shape
        assert result['fancy_indexed'].shape == (5, 5)
    
    def test_advanced_broadcasting(self):
        """Test broadcasting operations."""
        result = self.arrays.advanced_broadcasting((3, 1, 2), (1, 4, 1))
        
        assert 'broadcasted_shape' in result
        assert result['broadcasted_shape'] == (3, 4, 2)
        
        assert 'outer_product' in result
        assert result['outer_product'].shape == (10, 15)
    
    def test_structured_arrays_demo(self):
        """Test structured array operations."""
        result = self.arrays.structured_arrays_demo()
        
        assert 'original_data' in result
        assert 'sorted_by_age' in result
        assert 'tall_people' in result
        assert 'bmi' in result
        assert 'average_scores' in result
        
        # Check data types
        assert result['original_data'].dtype.names is not None
        assert 'name' in result['original_data'].dtype.names
        assert 'age' in result['original_data'].dtype.names
        
        # Check BMI calculation
        assert len(result['bmi']) == len(result['original_data'])
        assert all(bmi > 0 for bmi in result['bmi'])
    
    def test_memory_views_and_strides(self):
        """Test memory view operations."""
        result = self.arrays.memory_views_and_strides((10, 10))
        
        assert 'memory_shared' in result
        assert result['memory_shared'] == True
        
        assert 'original_strides' in result
        assert 'transposed_strides' in result
        
        # Check stride relationship for transpose
        orig_strides = result['original_strides']
        trans_strides = result['transposed_strides']
        assert orig_strides == trans_strides[::-1]
    
    def test_vectorized_operations_demo(self):
        """Test vectorized operations."""
        result = self.arrays.vectorized_operations_demo()
        
        assert 'data_shape' in result
        assert 'correlation_matrix_shape' in result
        assert 'outlier_counts' in result
        
        # Check correlation matrix is square
        corr_shape = result['correlation_matrix_shape']
        assert corr_shape[0] == corr_shape[1]
        assert corr_shape[0] == result['data_shape'][1]
    
    def test_advanced_ufuncs(self):
        """Test universal function operations."""
        result = self.arrays.advanced_ufuncs()
        
        assert 'custom_ufunc_result' in result
        assert 'cumsum_shape' in result
        assert 'sin_cos_combined_shape' in result
        
        # Check ufunc result shape
        assert result['custom_ufunc_result'].ndim == 2