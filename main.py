#!/usr/bin/env python3
"""
Practical Task P2 - Advanced NumPy Toolkit
Main Entry Point

Author: George Dorochov
Email: jordanaftermidnight@gmail.com

This is the simplified main entry point that runs both tasks:
1. Simulation Proof: Dog breeder picture shredding effect
2. Data Analysis: NumPy operations with statistical analysis

Usage: python3 main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Optional

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n🔹 {title}")
    print("-" * 40)

class SimpleSimulationProof:
    """
    Simplified version of the simulation proof demonstration.
    Shows the "impossible" dog breeder picture shredding effect.
    """
    
    def __init__(self, size: tuple = (100, 150)):
        self.height, self.width = size
        
    def create_test_image(self) -> np.ndarray:
        """Create a simple test pattern image."""
        image = np.zeros((self.height, self.width, 3))
        
        # Create a simple dog-like silhouette
        center_x, center_y = self.width // 2, self.height // 3
        
        # Head (circle)
        y, x = np.ogrid[:self.height, :self.width]
        head_mask = (x - center_x)**2 + (y - center_y)**2 <= 25**2
        
        # Body (rectangle)
        body_mask = ((x >= center_x - 25) & (x <= center_x + 25) & 
                    (y >= center_y + 20) & (y <= center_y + 60))
        
        # Combine parts and set color
        dog_mask = head_mask | body_mask
        image[dog_mask] = [0.6, 0.4, 0.2]  # Brown color
        
        # Add background pattern
        for i in range(0, self.width, 15):
            image[:, i:i+2, :] = [0.9, 0.9, 0.9]
        
        return image
    
    def shred_image(self, image: np.ndarray, strip_width: int = 12) -> list:
        """Shred image into vertical strips."""
        strips = []
        for i in range(0, self.width, strip_width):
            end_idx = min(i + strip_width, self.width)
            strip = image[:, i:end_idx, :].copy()
            strips.append(strip)
        return strips
    
    def create_impossible_effect(self, strips: list) -> Dict[str, np.ndarray]:
        """Create the 'impossible' multiplication effect."""
        results = {}
        
        # Normal reconstruction
        normal = np.concatenate(strips, axis=1)
        results['normal'] = normal
        
        # "Impossible" doubled reconstruction
        doubled_strips = []
        for strip in strips:
            doubled_strips.append(strip)
            # Add slightly modified version
            modified = strip.copy() * 0.95  # Slightly darker
            doubled_strips.append(modified)
        
        doubled = np.concatenate(doubled_strips, axis=1)
        results['doubled'] = doubled
        
        return results
    
    def analyze_violation(self, original: np.ndarray, doubled: np.ndarray) -> dict:
        """Analyze the 'physics violations'."""
        return {
            'energy_original': np.sum(original),
            'energy_doubled': np.sum(doubled),
            'energy_ratio': np.sum(doubled) / np.sum(original),
            'impossible_creation': np.sum(doubled) > np.sum(original) * 1.5
        }

class SimpleDataAnalysis:
    """
    Simplified data analysis with all required NumPy operations.
    """
    
    def __init__(self):
        np.random.seed(42)  # For reproducible results
    
    def generate_realistic_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate realistic multi-domain data."""
        data = {}
        
        # Weather data for major cities
        cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Mumbai', 'Cairo', 'Toronto', 'Berlin']
        weather_temps = np.array([15, 12, 18, 22, 28, 25, 8, 10]) + np.random.normal(0, 5, len(cities))
        weather_humidity = np.random.uniform(40, 85, len(cities))
        weather_pressure = np.random.normal(1013, 15, len(cities))
        
        data['weather'] = {
            'temperatures': weather_temps,
            'humidity': weather_humidity, 
            'pressure': weather_pressure,
            'cities': cities
        }
        
        # Stock market data
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META']
        stock_prices = np.array([180, 125, 375, 145, 220, 310]) + np.random.normal(0, 20, len(stocks))
        stock_volumes = np.random.uniform(2e7, 8e7, len(stocks))
        stock_changes = np.random.normal(0, 3, len(stocks))
        
        data['stocks'] = {
            'prices': stock_prices,
            'volumes': stock_volumes,
            'daily_changes': stock_changes,
            'symbols': stocks
        }
        
        # Economic data
        countries = ['USA', 'China', 'Japan', 'Germany', 'UK', 'India']
        gdp_data = np.array([25.5, 17.8, 4.2, 4.1, 3.1, 3.4]) + np.random.normal(0, 0.5, len(countries))
        population_data = np.array([333, 1412, 125, 83, 67, 1408]) + np.random.normal(0, 10, len(countries))
        
        data['economics'] = {
            'gdp_trillion': gdp_data,
            'population_millions': population_data,
            'countries': countries
        }
        
        return data
    
    def perform_required_operations(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Perform all required NumPy operations."""
        results = {}
        
        # Collect all numeric data
        all_numeric = []
        data_info = []
        
        for category, datasets in data.items():
            for name, array in datasets.items():
                if isinstance(array, np.ndarray) and array.dtype.kind in 'biufc':
                    all_numeric.extend(array.tolist())
                    data_info.append(f"{category}_{name}")
        
        sample_data = np.array(all_numeric)
        
        # Required Operations
        results['mean'] = np.mean(sample_data)
        results['sum'] = np.sum(sample_data)
        results['std'] = np.std(sample_data)
        results['variance'] = np.var(sample_data)
        results['min'] = np.min(sample_data)
        results['max'] = np.max(sample_data)
        results['median'] = np.median(sample_data)
        
        # Array filtering
        above_mean = sample_data[sample_data > results['mean']]
        results['filtered_above_mean'] = len(above_mean)
        results['filter_percentage'] = len(above_mean) / len(sample_data) * 100
        
        # Outlier detection using IQR
        q1, q3 = np.percentile(sample_data, [25, 75])
        iqr = q3 - q1
        outlier_mask = (sample_data < q1 - 1.5*iqr) | (sample_data > q3 + 1.5*iqr)
        outliers = sample_data[outlier_mask]
        results['outliers_count'] = len(outliers)
        results['outliers_percentage'] = len(outliers) / len(sample_data) * 100
        
        # Additional statistics
        results['percentiles'] = {
            '25th': q1,
            '50th': np.percentile(sample_data, 50),
            '75th': q3,
            '90th': np.percentile(sample_data, 90)
        }
        
        results['range'] = results['max'] - results['min']
        results['total_data_points'] = len(sample_data)
        results['data_categories'] = len(data)
        
        return results
    
    def analyze_correlations(self, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze correlations between datasets."""
        # Get weather data for correlation analysis
        weather = data['weather']
        temp_vs_humidity = np.corrcoef(weather['temperatures'], weather['humidity'])[0, 1]
        temp_vs_pressure = np.corrcoef(weather['temperatures'], weather['pressure'])[0, 1]
        
        # Get stock data correlations
        stocks = data['stocks']
        price_vs_volume = np.corrcoef(stocks['prices'], stocks['volumes'])[0, 1]
        
        return {
            'temp_humidity_correlation': temp_vs_humidity,
            'temp_pressure_correlation': temp_vs_pressure,
            'price_volume_correlation': price_vs_volume,
            'strongest_correlation': max(abs(temp_vs_humidity), abs(temp_vs_pressure), abs(price_vs_volume))
        }

def create_visualizations(sim_results: dict, data_results: dict, analysis: dict):
    """Create simplified but informative visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Practical Task P2 - Results Summary\nAuthor: George Dorochov', fontsize=14, fontweight='bold')
    
    # Simulation proof
    axes[0, 0].imshow(sim_results['doubled'])
    axes[0, 0].set_title('Task 1: "Impossible" Image Doubling', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Energy violation
    energy_data = [sim_results['analysis']['energy_original'], sim_results['analysis']['energy_doubled']]
    axes[0, 1].bar(['Original', 'Doubled'], energy_data, color=['blue', 'red'])
    axes[0, 1].set_title('Energy Conservation Violation!', fontweight='bold', color='red')
    axes[0, 1].set_ylabel('Total Energy')
    
    # Data statistics
    stats_names = ['Mean', 'Median', 'Std', 'Range']
    stats_values = [analysis['mean'], analysis['median'], analysis['std'], analysis['range']]
    axes[1, 0].bar(stats_names, stats_values)
    axes[1, 0].set_title('Task 2: Statistical Analysis')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Data filtering results
    filter_data = [analysis['total_data_points'] - analysis['filtered_above_mean'], 
                   analysis['filtered_above_mean']]
    axes[1, 1].pie(filter_data, labels=['Below Mean', 'Above Mean'], autopct='%1.1f%%')
    axes[1, 1].set_title('Array Filtering Results')
    
    plt.tight_layout()
    plt.savefig('practical_task_p2_results.png', dpi=200, bbox_inches='tight')
    print("📊 Visualization saved as 'practical_task_p2_results.png'")
    
    return fig

def main():
    """Main execution function."""
    print_header("PRACTICAL TASK P2 - ADVANCED NUMPY TOOLKIT")
    print("Author: George Dorochov")
    print("Email: jordanaftermidnight@gmail.com")
    print("\nDemonstrating:")
    print("1. Simulation Proof: Dog breeder picture shredding effect")
    print("2. Data Analysis: Comprehensive NumPy operations")
    
    # ============ TASK 1: SIMULATION PROOF ============
    print_header("TASK 1: SIMULATION PROOF")
    
    sim_proof = SimpleSimulationProof()
    
    print_section("Creating and Shredding Image")
    original_image = sim_proof.create_test_image()
    print(f"✅ Created test image: {original_image.shape}")
    
    strips = sim_proof.shred_image(original_image)
    print(f"✅ Shredded into {len(strips)} strips")
    
    print_section("Generating 'Impossible' Effects")
    effects = sim_proof.create_impossible_effect(strips)
    print(f"✅ Normal reconstruction: {effects['normal'].shape}")
    print(f"✅ Impossible doubling: {effects['doubled'].shape}")
    
    analysis = sim_proof.analyze_violation(effects['normal'], effects['doubled'])
    print(f"\n🔬 Physics Violation Analysis:")
    print(f"   Original energy: {analysis['energy_original']:.2f}")
    print(f"   Doubled energy:  {analysis['energy_doubled']:.2f}")
    print(f"   Energy ratio:    {analysis['energy_ratio']:.2f}x")
    
    if analysis['impossible_creation']:
        print("   ⚠️  VIOLATION: Energy created from nothing!")
        print("   🤖 Evidence we live in a simulation!")
    
    sim_results = {'doubled': effects['doubled'], 'analysis': analysis}
    
    # ============ TASK 2: DATA ANALYSIS ============
    print_header("TASK 2: DATA ANALYSIS WITH NUMPY")
    
    analyzer = SimpleDataAnalysis()
    
    print_section("Generating Multi-Domain Data")
    data = analyzer.generate_realistic_data()
    
    total_datasets = sum(len([k for k, v in datasets.items() if isinstance(v, np.ndarray)]) 
                        for datasets in data.values())
    total_points = sum(len(v) for datasets in data.values() 
                      for k, v in datasets.items() if isinstance(v, np.ndarray))
    
    print(f"✅ Generated {len(data)} data categories")
    print(f"✅ Total datasets: {total_datasets}")
    print(f"✅ Total data points: {total_points}")
    
    for category, datasets in data.items():
        numeric_count = sum(1 for k, v in datasets.items() if isinstance(v, np.ndarray))
        print(f"   {category}: {numeric_count} datasets")
    
    print_section("Performing Required NumPy Operations")
    analysis = analyzer.perform_required_operations(data)
    
    # Display all required operations
    print("📊 DESCRIPTIVE STATISTICS:")
    print(f"   Mean:     {analysis['mean']:.4f}")
    print(f"   Sum:      {analysis['sum']:.4f}")
    print(f"   Std Dev:  {analysis['std']:.4f}")
    print(f"   Variance: {analysis['variance']:.4f}")
    print(f"   Min:      {analysis['min']:.4f}")
    print(f"   Max:      {analysis['max']:.4f}")
    print(f"   Median:   {analysis['median']:.4f}")
    print(f"   Range:    {analysis['range']:.4f}")
    
    print(f"\n🎯 ARRAY FILTERING:")
    print(f"   Values above mean: {analysis['filtered_above_mean']} ({analysis['filter_percentage']:.1f}%)")
    print(f"   Outliers detected: {analysis['outliers_count']} ({analysis['outliers_percentage']:.1f}%)")
    
    print(f"\n📈 PERCENTILES:")
    for percentile, value in analysis['percentiles'].items():
        print(f"   {percentile}: {value:.4f}")
    
    print_section("Correlation Analysis")
    correlations = analyzer.analyze_correlations(data)
    print(f"✅ Temperature vs Humidity: {correlations['temp_humidity_correlation']:.3f}")
    print(f"✅ Temperature vs Pressure: {correlations['temp_pressure_correlation']:.3f}")
    print(f"✅ Stock Price vs Volume:   {correlations['price_volume_correlation']:.3f}")
    print(f"✅ Strongest correlation:   {correlations['strongest_correlation']:.3f}")
    
    # ============ VISUALIZATION ============
    print_header("GENERATING VISUALIZATIONS")
    create_visualizations(sim_results, data, analysis)
    
    # ============ SUMMARY ============
    print_header("EXECUTION SUMMARY")
    print("✅ TASK 1 COMPLETED: Simulation proof demonstrates 'impossible' physics violations")
    print("✅ TASK 2 COMPLETED: Comprehensive NumPy analysis with all required operations")
    print(f"\n📊 NumPy Operations Verified:")
    print(f"   ✓ Mean/Average calculations")
    print(f"   ✓ Sum operations") 
    print(f"   ✓ Descriptive statistics (std, var, min, max, median)")
    print(f"   ✓ Array filtering (boolean indexing, outlier detection)")
    print(f"   ✓ Advanced operations (percentiles, correlations)")
    
    print(f"\n📈 Data Processing Summary:")
    print(f"   • {total_datasets} datasets analyzed")
    print(f"   • {total_points} data points processed")
    print(f"   • {len(data)} data categories (weather, stocks, economics)")
    print(f"   • Multiple filtering methods applied")
    print(f"   • Correlation analysis performed")
    
    print(f"\n🎯 Project Status: COMPLETE")
    print(f"   Author: George Dorochov")
    print(f"   Email: jordanaftermidnight@gmail.com")
    print(f"   All requirements satisfied!")

if __name__ == "__main__":
    main()