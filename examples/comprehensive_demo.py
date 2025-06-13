"""
Comprehensive demonstration of advanced NumPy features.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_arrays import AdvancedArrays
from signal_processing import SignalProcessor
from linear_algebra import MatrixOperations

def run_advanced_arrays_demo():
    """Demonstrate advanced array operations."""
    print("=" * 60)
    print("ADVANCED ARRAY OPERATIONS DEMO")
    print("=" * 60)
    
    arrays = AdvancedArrays()
    
    # Fancy indexing demonstration
    print("\n1. Fancy Indexing and Boolean Operations")
    result = arrays.fancy_indexing_demo((50, 50))
    print(f"Original shape: {result['original_shape']}")
    print(f"Boolean selected elements: {len(result['boolean_selected'])}")
    print(f"Fancy indexed shape: {result['fancy_indexed'].shape}")
    print(f"Sliced shape: {result['sliced_shape']}")
    
    # Broadcasting demonstration
    print("\n2. Advanced Broadcasting")
    broadcast_result = arrays.advanced_broadcasting()
    print(f"Input shapes: {broadcast_result['a_shape']} and {broadcast_result['b_shape']}")
    print(f"Broadcasted result shape: {broadcast_result['broadcasted_shape']}")
    print(f"Outer product shape: {broadcast_result['outer_product'].shape}")
    
    # Structured arrays
    print("\n3. Structured Arrays")
    struct_result = arrays.structured_arrays_demo()
    print(f"Data shape: {struct_result['original_data'].shape}")
    print(f"Average BMI: {np.mean(struct_result['bmi']):.2f}")
    print(f"Tall people count: {len(struct_result['tall_people'])}")
    
    # Memory views and strides
    print("\n4. Memory Views and Strides")
    memory_result = arrays.memory_views_and_strides((100, 100))
    print(f"Original strides: {memory_result['original_strides']}")
    print(f"Transposed strides: {memory_result['transposed_strides']}")
    print(f"Memory shared with transpose: {memory_result['memory_shared']}")
    
    # Vectorized operations
    print("\n5. Vectorized Operations")
    vec_result = arrays.vectorized_operations_demo()
    print(f"Processed {vec_result['data_shape'][0]} samples with {vec_result['data_shape'][1]} features")
    print(f"Correlation matrix shape: {vec_result['correlation_matrix_shape']}")
    print(f"Outliers per feature: {vec_result['outlier_counts']}")

def run_signal_processing_demo():
    """Demonstrate signal processing capabilities."""
    print("\n" + "=" * 60)
    print("SIGNAL PROCESSING DEMO")
    print("=" * 60)
    
    processor = SignalProcessor(sampling_rate=1000)
    
    # Generate complex signal
    print("\n1. Signal Generation and Analysis")
    t, signal_data = processor.generate_complex_signal(
        duration=2.0,
        frequencies=[5, 25, 50, 75],
        amplitudes=[1.0, 0.8, 0.6, 0.4],
        noise_level=0.2
    )
    print(f"Generated signal: {len(signal_data)} samples over {t[-1]:.1f} seconds")
    
    # Spectral analysis
    print("\n2. Spectral Analysis")
    spectral_result = processor.spectral_analysis(signal_data)
    print(f"Frequency resolution: {spectral_result['frequencies'][1] - spectral_result['frequencies'][0]:.2f} Hz")
    print(f"Detected peaks at: {spectral_result['peaks']['frequencies'][:5]} Hz")
    print(f"Spectral centroid: {spectral_result['spectral_features']['centroid']:.2f} Hz")
    
    # Advanced filtering
    print("\n3. Advanced Filtering")
    filter_results = processor.advanced_filtering(signal_data)
    print(f"Applied {len(filter_results)} different filters:")
    for filter_name in filter_results.keys():
        print(f"  - {filter_name}")
    
    # Wavelet analysis
    print("\n4. Wavelet Analysis")
    wavelet_result = processor.wavelets_analysis(signal_data[:1000])  # Shorter for demo
    print(f"Wavelet transform shape: {wavelet_result['scalogram'].shape}")
    print(f"Frequency range: {wavelet_result['frequencies'][0]:.1f} - {wavelet_result['frequencies'][-1]:.1f} Hz")
    
    # Nonlinear analysis
    print("\n5. Nonlinear Analysis")
    nonlinear_result = processor.nonlinear_analysis(signal_data[:1000])
    print(f"Number of EMD components: {nonlinear_result['emd']['n_imfs']}")
    print(f"Estimated fractal dimension: {nonlinear_result['fractal_analysis']['dimension']:.3f}")

def run_linear_algebra_demo():
    """Demonstrate linear algebra operations."""
    print("\n" + "=" * 60)
    print("LINEAR ALGEBRA DEMO")
    print("=" * 60)
    
    linalg = MatrixOperations()
    
    # Create test matrices
    print("\n1. Matrix Creation and Properties")
    np.random.seed(42)
    A = np.random.randn(50, 50)
    A = A + A.T  # Make symmetric
    b = np.random.randn(50)
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Vector b shape: {b.shape}")
    
    # Matrix analysis
    print("\n2. Matrix Analysis")
    analysis_result = linalg.advanced_matrix_analysis(A)
    print(f"Matrix properties:")
    for prop, value in analysis_result['properties'].items():
        print(f"  {prop}: {value}")
    print(f"Condition number (L2): {analysis_result['condition_numbers']['l2']:.2e}")
    print(f"Numerical rank: {analysis_result['rank_analysis']['numerical_rank']}")
    
    # Matrix decompositions
    print("\n3. Matrix Decompositions")
    decomp_result = linalg.matrix_decompositions(A)
    print("Available decompositions:")
    for decomp_name, decomp_data in decomp_result.items():
        if 'error' not in decomp_data:
            verification = decomp_data.get('verification', 'N/A')
            print(f"  {decomp_name}: Verified = {verification}")
        else:
            print(f"  {decomp_name}: Error occurred")
    
    # Linear system solving
    print("\n4. Linear System Solutions")
    systems_result = linalg.advanced_linear_systems(A, b)
    print("Solution methods:")
    for method_name, method_data in systems_result.items():
        if 'error' not in method_data:
            residual = method_data.get('residual', 'N/A')
            print(f"  {method_name}: Residual = {residual}")
        else:
            print(f"  {method_name}: Error occurred")
    
    # Matrix functions
    print("\n5. Matrix Functions")
    functions_result = linalg.matrix_functions(A[:5, :5])  # Smaller matrix for demo
    print("Matrix functions computed:")
    for func_name, func_data in functions_result.items():
        if 'error' not in func_data:
            if func_name == 'powers':
                print(f"  {func_name}: {list(func_data.keys())}")
            else:
                print(f"  {func_name}: Success")
        else:
            print(f"  {func_name}: Error occurred")

def create_visualization():
    """Create some visualizations of the results."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Signal processing visualization
    processor = SignalProcessor()
    t, signal_data = processor.generate_complex_signal(duration=1.0)
    spectral_result = processor.spectral_analysis(signal_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain signal
    axes[0, 0].plot(t[:500], signal_data[:500])
    axes[0, 0].set_title('Time Domain Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Frequency domain
    axes[0, 1].semilogy(spectral_result['frequencies'], spectral_result['psd'])
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power')
    
    # Spectrogram
    spec_data = spectral_result['spectrogram']
    im = axes[1, 0].pcolormesh(
        spec_data['times'], 
        spec_data['frequencies'], 
        spec_data['magnitude'],
        shading='auto'
    )
    axes[1, 0].set_title('Spectrogram')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[1, 0], label='Magnitude (dB)')
    
    # Matrix visualization
    A = np.random.randn(20, 20)
    U, s, Vt = np.linalg.svd(A)
    
    axes[1, 1].semilogy(s, 'o-')
    axes[1, 1].set_title('Singular Values')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Singular Value')
    
    plt.tight_layout()
    plt.savefig('/Users/jordan_after_midnight/Practical-Task-P2/examples/demo_results.png', dpi=150)
    print("Visualization saved as 'demo_results.png'")

def main():
    """Run all demonstrations."""
    print("Advanced NumPy Toolkit - Comprehensive Demonstration")
    print("Practical Task P2")
    print("=" * 60)
    
    try:
        run_advanced_arrays_demo()
        run_signal_processing_demo()
        run_linear_algebra_demo()
        create_visualization()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nAll advanced NumPy features have been demonstrated.")
        print("Check the generated visualization for graphical results.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()