"""
Complete Demonstration: Simulation Proof + Data Analysis

This script combines both tasks:
1. "Prove we live in a simulation" using the dog breeder picture shredding effect
2. Web scraping with NumPy data analysis including descriptive statistics and filtering

Run this script to see both demonstrations in action!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing import SimulationProof
from data_analysis import WebScraper, NumpyAnalytics

def task_1_simulation_proof():
    """
    Task 1: Recreate the dog breeder picture shredding effect to 'prove' we live in a simulation.
    """
    print("\n" + "🤖" * 30)
    print("TASK 1: PROVING WE LIVE IN A SIMULATION")
    print("The Dog Breeder Picture Shredding Effect")
    print("🤖" * 30)
    
    # Initialize the simulation proof system
    sim_proof = SimulationProof(image_size=(200, 300))
    
    print("\n🔬 Creating test image (simulated dog photo)...")
    original_image = sim_proof.create_test_image("dog_silhouette")
    
    print("✂️  Shredding image into strips (like a paper shredder)...")
    strips = sim_proof.shred_image(strip_width=15)
    print(f"   Created {len(strips)} strips from the original image")
    
    print("🎭 Demonstrating 'impossible' reconstruction effects...")
    
    # Create the full simulation proof sequence
    sequence = sim_proof.create_simulation_proof_sequence()
    
    print("\n📊 Analyzing the 'evidence' for simulation...")
    analysis = sim_proof.analyze_simulation_evidence(sequence)
    
    # Print the simulation report
    sim_proof.print_simulation_report(analysis)
    
    # Create visualization
    print("\n📈 Creating visualization of simulation proof...")
    fig = sim_proof.visualize_proof(sequence, 
                                   save_path='/Users/jordan_after_midnight/Practical-Task-P2/examples/simulation_proof.png')
    
    return sequence, analysis

def task_2_data_analysis():
    """
    Task 2: Web scraping with NumPy data analysis.
    """
    print("\n" + "📊" * 30)
    print("TASK 2: WEB SCRAPING + NUMPY DATA ANALYSIS")
    print("📊" * 30)
    
    # Initialize scraper and analytics
    scraper = WebScraper(delay=0.1)  # Faster for demo
    analytics = NumpyAnalytics()
    
    # Collect data from multiple sources
    print("\n🌐 Collecting data from multiple sources...")
    all_data = scraper.collect_all_data()
    
    # Display collected data summary
    print("\n📋 Data Collection Summary:")
    for category, datasets in all_data.items():
        print(f"   {category.upper()}:")
        for data_name, data_array in datasets.items():
            if isinstance(data_array, np.ndarray):
                print(f"     {data_name}: {data_array.shape} - {data_array.dtype}")
            else:
                print(f"     {data_name}: {type(data_array)} (metadata)")
    
    # Perform comprehensive analysis
    print("\n🔬 Performing comprehensive NumPy analysis...")
    analysis_results = analytics.comprehensive_analysis(all_data)
    
    # Print detailed analysis report
    analytics.print_analysis_report(analysis_results)
    
    # Create visualizations
    print("\n📈 Creating analysis visualizations...")
    fig = analytics.create_visualizations(all_data, analysis_results,
                                        save_path='/Users/jordan_after_midnight/Practical-Task-P2/examples/data_analysis.png')
    
    # Demonstrate specific NumPy operations
    demonstrate_numpy_operations(all_data)
    
    return all_data, analysis_results

def demonstrate_numpy_operations(scraped_data):
    """
    Demonstrate specific NumPy operations required by the task.
    """
    print("\n" + "⚡" * 40)
    print("DETAILED NUMPY OPERATIONS DEMONSTRATION")
    print("⚡" * 40)
    
    # Find some numeric data to work with
    sample_data = None
    data_name = ""
    
    for category, datasets in scraped_data.items():
        for name, data_array in datasets.items():
            if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                sample_data = data_array
                data_name = f"{category}_{name}"
                break
        if sample_data is not None:
            break
    
    if sample_data is None:
        print("No numeric data found for demonstration")
        return
    
    print(f"\n📊 Working with: {data_name}")
    print(f"   Shape: {sample_data.shape}")
    print(f"   Data type: {sample_data.dtype}")
    print(f"   Sample values: {sample_data[:5]}")
    
    print(f"\n🧮 BASIC DESCRIPTIVE STATISTICS:")
    print(f"   Mean: {np.mean(sample_data):.4f}")
    print(f"   Median: {np.median(sample_data):.4f}")
    print(f"   Standard Deviation: {np.std(sample_data):.4f}")
    print(f"   Variance: {np.var(sample_data):.4f}")
    print(f"   Minimum: {np.min(sample_data):.4f}")
    print(f"   Maximum: {np.max(sample_data):.4f}")
    print(f"   Sum: {np.sum(sample_data):.4f}")
    print(f"   Range: {np.max(sample_data) - np.min(sample_data):.4f}")
    
    print(f"\n🎯 PERCENTILES:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(sample_data, p)
        print(f"   {p}th percentile: {value:.4f}")
    
    print(f"\n🔍 ARRAY FILTERING DEMONSTRATIONS:")
    
    # 1. Filter values above mean
    above_mean = sample_data[sample_data > np.mean(sample_data)]
    print(f"   Values above mean: {len(above_mean)} out of {len(sample_data)} ({len(above_mean)/len(sample_data)*100:.1f}%)")
    
    # 2. Filter outliers using IQR method
    q1, q3 = np.percentile(sample_data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = sample_data[(sample_data < lower_bound) | (sample_data > upper_bound)]
    filtered_data = sample_data[(sample_data >= lower_bound) & (sample_data <= upper_bound)]
    
    print(f"   IQR-based outliers: {len(outliers)} removed, {len(filtered_data)} remaining")
    print(f"   Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # 3. Filter top and bottom 10%
    p10 = np.percentile(sample_data, 10)
    p90 = np.percentile(sample_data, 90)
    middle_80_percent = sample_data[(sample_data >= p10) & (sample_data <= p90)]
    
    print(f"   Middle 80% filter: {len(middle_80_percent)} values in range [{p10:.4f}, {p90:.4f}]")
    
    # 4. Boolean filtering with conditions
    positive_values = sample_data[sample_data > 0]
    high_values = sample_data[sample_data > np.mean(sample_data) + np.std(sample_data)]
    
    print(f"   Positive values: {len(positive_values)} out of {len(sample_data)}")
    print(f"   High values (> mean + std): {len(high_values)} out of {len(sample_data)}")
    
    print(f"\n⚡ ADVANCED ARRAY OPERATIONS:")
    
    # Cumulative operations
    cumsum = np.cumsum(sample_data)
    cumprod = np.cumprod(sample_data[:5])  # Limit cumprod to avoid overflow
    
    print(f"   Cumulative sum range: [{cumsum[0]:.4f}, {cumsum[-1]:.4f}]")
    print(f"   First 5 cumulative products: {cumprod}")
    
    # Sorting and ranking
    sorted_data = np.sort(sample_data)
    argsort_indices = np.argsort(sample_data)
    
    print(f"   Sorted range: [{sorted_data[0]:.4f}, {sorted_data[-1]:.4f}]")
    print(f"   Index of minimum value: {argsort_indices[0]}")
    print(f"   Index of maximum value: {argsort_indices[-1]}")
    
    # Mathematical operations
    log_data = np.log(np.abs(sample_data) + 1)  # Add 1 to handle zeros/negatives
    sqrt_data = np.sqrt(np.abs(sample_data))
    squared_data = np.square(sample_data)
    
    print(f"   Log-transformed mean: {np.mean(log_data):.4f}")
    print(f"   Square-root mean: {np.mean(sqrt_data):.4f}")
    print(f"   Squared mean: {np.mean(squared_data):.4f}")
    
    # Cross-dataset analysis if we have multiple arrays
    all_numeric_arrays = []
    all_names = []
    
    for category, datasets in scraped_data.items():
        for name, data_array in datasets.items():
            if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                all_numeric_arrays.append(data_array)
                all_names.append(f"{category}_{name}")
    
    if len(all_numeric_arrays) > 1:
        print(f"\n🔗 CROSS-DATASET ANALYSIS:")
        
        # Find minimum length for correlation analysis
        min_length = min(len(arr) for arr in all_numeric_arrays)
        aligned_arrays = [arr[:min_length] for arr in all_numeric_arrays]
        
        # Create correlation matrix
        data_matrix = np.column_stack(aligned_arrays)
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        print(f"   Correlation matrix shape: {correlation_matrix.shape}")
        
        # Find highest correlation (excluding diagonal)
        np.fill_diagonal(correlation_matrix, 0)  # Remove self-correlations
        max_corr_idx = np.unravel_index(np.argmax(np.abs(correlation_matrix)), correlation_matrix.shape)
        max_corr_value = correlation_matrix[max_corr_idx]
        
        print(f"   Highest correlation: {max_corr_value:.4f}")
        print(f"   Between: {all_names[max_corr_idx[0]]} and {all_names[max_corr_idx[1]]}")
        
        # Calculate means of all datasets
        all_means = [np.mean(arr) for arr in all_numeric_arrays]
        print(f"   Mean of all means: {np.mean(all_means):.4f}")
        print(f"   Standard deviation of means: {np.std(all_means):.4f}")

def create_combined_visualization(sim_sequence, sim_analysis, data_results):
    """
    Create a combined visualization showing both tasks.
    """
    print("\n📊 Creating combined visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Simulation proof subplot
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.imshow(sim_sequence['step_4_doubled'])
    ax1.set_title('SIMULATION PROOF: Impossible Image Multiplication', fontweight='bold', color='red')
    ax1.axis('off')
    
    # Energy conservation violation
    ax2 = plt.subplot(2, 3, 3)
    original_energy = sim_analysis['pixel_conservation']['original_total_brightness']
    doubled_energy = sim_analysis['pixel_conservation']['doubled_total_brightness']
    
    ax2.bar(['Original', 'Doubled'], [original_energy, doubled_energy], color=['blue', 'red'])
    ax2.set_title('Energy Conservation VIOLATED!', fontweight='bold', color='red')
    ax2.set_ylabel('Total Brightness Energy')
    
    # Data analysis results
    if data_results:
        # Sample data distribution
        ax3 = plt.subplot(2, 3, 4)
        for category, datasets in data_results.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                    ax3.hist(data_array, bins=20, alpha=0.7, label=f"{category}_{data_name}")
                    break
            break
        ax3.set_title('Sample Data Distribution')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Means comparison
        ax4 = plt.subplot(2, 3, 5)
        means = []
        labels = []
        for category, datasets in data_results.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                    means.append(np.mean(data_array))
                    labels.append(f"{category}_{data_name}")
        
        if means:
            ax4.bar(range(len(means)), means)
            ax4.set_title('Mean Values Across Datasets')
            ax4.set_ylabel('Mean Value')
        
        # Standard deviations
        ax5 = plt.subplot(2, 3, 6)
        stds = []
        for category, datasets in data_results.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                    stds.append(np.std(data_array))
        
        if stds:
            ax5.bar(range(len(stds)), stds)
            ax5.set_title('Standard Deviations')
            ax5.set_ylabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig('/Users/jordan_after_midnight/Practical-Task-P2/examples/combined_results.png', 
                dpi=300, bbox_inches='tight')
    print("Combined visualization saved as 'combined_results.png'")
    
    return fig

def main():
    """
    Main function that runs both tasks.
    """
    print("🚀" * 50)
    print("PRACTICAL TASK P2 - COMPLETE DEMONSTRATION")
    print("Advanced NumPy: Simulation Proof + Data Analysis")
    print("🚀" * 50)
    
    try:
        # Task 1: Simulation Proof
        sim_sequence, sim_analysis = task_1_simulation_proof()
        
        # Task 2: Data Analysis
        scraped_data, analysis_results = task_2_data_analysis()
        
        # Combined visualization
        create_combined_visualization(sim_sequence, sim_analysis, scraped_data)
        
        print("\n" + "✅" * 30)
        print("BOTH TASKS COMPLETED SUCCESSFULLY!")
        print("✅" * 30)
        
        print("\n📁 Generated Files:")
        print("   📊 simulation_proof.png - Visualization of the 'simulation glitch'")
        print("   📈 data_analysis.png - Comprehensive data analysis results")
        print("   🔗 combined_results.png - Combined visualization of both tasks")
        
        print("\n🎯 Summary:")
        print("   ✓ Task 1: Successfully demonstrated the 'simulation proof' effect")
        print("   ✓ Task 2: Scraped data and performed comprehensive NumPy analysis")
        print("   ✓ Implemented all required operations: mean, sum, filtering, statistics")
        print("   ✓ Created comprehensive documentation and visualizations")
        
        print("\n🔬 Key NumPy Features Demonstrated:")
        print("   • Advanced array operations and broadcasting")
        print("   • Image processing and manipulation")
        print("   • Statistical analysis and descriptive statistics")
        print("   • Data filtering and outlier detection")
        print("   • Correlation analysis and matrix operations")
        print("   • Fourier transforms and signal processing")
        print("   • Vectorized operations and performance optimization")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()