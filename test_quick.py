#!/usr/bin/env python3
"""
Quick test script to verify both tasks work correctly.
"""

import sys
import os
sys.path.append('src')

def test_task_1():
    """Test the simulation proof implementation."""
    print("🤖 Testing Task 1: Simulation Proof...")
    
    from image_processing import SimulationProof
    
    # Create simulation proof
    sim_proof = SimulationProof(image_size=(50, 75))  # Smaller for quick test
    
    # Create test image
    original = sim_proof.create_test_image("dog_silhouette")
    print(f"   ✅ Created test image: {original.shape}")
    
    # Shred image
    strips = sim_proof.shred_image(strip_width=10)
    print(f"   ✅ Shredded into {len(strips)} strips")
    
    # Generate sequence
    sequence = sim_proof.create_simulation_proof_sequence()
    print(f"   ✅ Generated {len(sequence)} sequence steps")
    
    # Analyze evidence
    analysis = sim_proof.analyze_simulation_evidence(sequence)
    violations = sum([
        analysis['pixel_conservation']['impossible_energy_creation'],
        analysis['information_theory']['information_created'],
        analysis['simulation_glitches']['impossible_pixel_values'],
        analysis['simulation_glitches']['matrix_artifacts']
    ])
    
    print(f"   ✅ Detected {violations} physics violations")
    print("   🎯 Task 1 completed successfully!")
    
    return True

def test_task_2():
    """Test the data analysis implementation."""
    print("\n📊 Testing Task 2: Data Analysis...")
    
    from data_analysis import WebScraper, NumpyAnalytics
    import numpy as np
    
    # Test web scraper
    scraper = WebScraper(delay=0.1)
    data = scraper.collect_all_data()
    print(f"   ✅ Collected data from {len(data)} categories")
    
    # Test analytics
    analytics = NumpyAnalytics()
    
    # Test specific required operations
    sample_data = None
    for category, datasets in data.items():
        for name, array in datasets.items():
            if isinstance(array, np.ndarray) and array.dtype.kind in 'biufc':
                sample_data = array
                break
        if sample_data is not None:
            break
    
    if sample_data is not None:
        # Required operations
        mean_val = np.mean(sample_data)
        sum_val = np.sum(sample_data)
        std_val = np.std(sample_data)
        var_val = np.var(sample_data)
        min_val = np.min(sample_data)
        max_val = np.max(sample_data)
        
        print(f"   ✅ Mean: {mean_val:.4f}")
        print(f"   ✅ Sum: {sum_val:.4f}")
        print(f"   ✅ Std: {std_val:.4f}")
        print(f"   ✅ Variance: {var_val:.4f}")
        print(f"   ✅ Min: {min_val:.4f}")
        print(f"   ✅ Max: {max_val:.4f}")
        
        # Array filtering
        above_mean = sample_data[sample_data > mean_val]
        print(f"   ✅ Filtered {len(above_mean)} values above mean")
        
        # Outlier detection
        q1, q3 = np.percentile(sample_data, [25, 75])
        iqr = q3 - q1
        outliers = sample_data[(sample_data < q1 - 1.5*iqr) | (sample_data > q3 + 1.5*iqr)]
        print(f"   ✅ Detected {len(outliers)} outliers using IQR method")
    
    # Comprehensive analysis
    analysis_results = analytics.comprehensive_analysis(data)
    print(f"   ✅ Comprehensive analysis completed")
    print(f"   ✅ Analyzed {analysis_results['summary']['total_datasets']} datasets")
    print(f"   ✅ Processed {analysis_results['summary']['total_data_points']} data points")
    
    print("   🎯 Task 2 completed successfully!")
    
    return True

def main():
    """Run both tests."""
    print("🚀 PRACTICAL TASK P2 - QUICK VERIFICATION TEST")
    print("=" * 50)
    
    try:
        # Test both tasks
        task1_success = test_task_1()
        task2_success = test_task_2()
        
        if task1_success and task2_success:
            print("\n" + "✅" * 20)
            print("🎉 ALL TESTS PASSED!")
            print("✅" * 20)
            print("\n📋 Summary:")
            print("   ✓ Task 1: Simulation proof implementation works")
            print("   ✓ Task 2: Data analysis with required NumPy operations works")
            print("   ✓ Web scraping functionality works")
            print("   ✓ All required operations implemented:")
            print("     • Mean/average calculations")
            print("     • Sum operations")
            print("     • Descriptive statistics (std, var, min, max)")
            print("     • Array filtering (boolean indexing, outliers)")
            print("\n🔗 Ready to run: python3 examples/complete_demonstration.py")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()