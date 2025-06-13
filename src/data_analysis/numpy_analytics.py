"""
NumPy Analytics Module

Advanced statistical analysis and data processing using NumPy arrays.
Performs comprehensive analysis on scraped data including descriptive statistics,
filtering, correlations, and advanced mathematical operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings

class NumpyAnalytics:
    """
    Advanced NumPy-based analytics for scraped data.
    """
    
    def __init__(self):
        self.analysis_results = {}
        
    def descriptive_statistics(self, data: np.ndarray, label: str = "Data") -> Dict[str, float]:
        """
        Calculate comprehensive descriptive statistics for a NumPy array.
        """
        if data.size == 0:
            return {"error": "Empty array"}
        
        stats_dict = {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else np.inf
        }
        
        return stats_dict
    
    def array_filtering(self, data: np.ndarray, filter_type: str = "outliers", 
                       threshold: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Apply various filtering techniques to NumPy arrays.
        """
        results = {}
        
        if filter_type == "outliers":
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            
            results['filtered_data'] = data[~outlier_mask]
            results['outliers'] = data[outlier_mask]
            results['outlier_indices'] = np.where(outlier_mask)[0]
            results['outlier_count'] = np.sum(outlier_mask)
            
        elif filter_type == "iqr":
            # Interquartile range based filtering
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (data >= lower_bound) & (data <= upper_bound)
            results['filtered_data'] = data[mask]
            results['outliers'] = data[~mask]
            results['bounds'] = {'lower': lower_bound, 'upper': upper_bound}
            
        elif filter_type == "percentile":
            # Percentile-based filtering
            lower_percentile = (100 - threshold) / 2
            upper_percentile = 100 - lower_percentile
            
            lower_bound = np.percentile(data, lower_percentile)
            upper_bound = np.percentile(data, upper_percentile)
            
            mask = (data >= lower_bound) & (data <= upper_bound)
            results['filtered_data'] = data[mask]
            results['outliers'] = data[~mask]
            results['bounds'] = {'lower': lower_bound, 'upper': upper_bound}
            
        elif filter_type == "moving_average":
            # Moving average smoothing
            window_size = int(threshold)
            if window_size >= len(data):
                window_size = len(data) // 2
            
            padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
            smoothed = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
            
            results['filtered_data'] = smoothed
            results['original_data'] = data
            results['window_size'] = window_size
        
        return results
    
    def correlation_analysis(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform correlation analysis between multiple datasets.
        """
        # Ensure all arrays have the same length
        min_length = min(len(arr) for arr in datasets.values())
        aligned_data = {key: arr[:min_length] for key, arr in datasets.items()}
        
        # Create correlation matrix
        data_matrix = np.column_stack(list(aligned_data.values()))
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        # Find strongest correlations
        labels = list(aligned_data.keys())
        correlations = []
        
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                corr_value = correlation_matrix[i, j]
                correlations.append({
                    'variables': (labels[i], labels[j]),
                    'correlation': corr_value,
                    'strength': self._correlation_strength(abs(corr_value))
                })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix,
            'labels': labels,
            'top_correlations': correlations[:5],
            'all_correlations': correlations
        }
    
    def _correlation_strength(self, corr: float) -> str:
        """Categorize correlation strength."""
        if corr >= 0.8:
            return "Very Strong"
        elif corr >= 0.6:
            return "Strong"
        elif corr >= 0.4:
            return "Moderate"
        elif corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def advanced_array_operations(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform advanced NumPy array operations and transformations.
        """
        results = {}
        
        # Fourier Transform Analysis
        if len(data) > 1:
            fft_result = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(data))
            
            results['fft'] = {
                'amplitudes': np.abs(fft_result),
                'phases': np.angle(fft_result),
                'frequencies': frequencies,
                'dominant_frequency_index': np.argmax(np.abs(fft_result[1:len(data)//2])) + 1
            }
        
        # Polynomial fitting
        if len(data) >= 3:
            x = np.arange(len(data))
            poly_coeffs = np.polyfit(x, data, deg=min(3, len(data)-1))
            poly_fit = np.polyval(poly_coeffs, x)
            
            results['polynomial_fit'] = {
                'coefficients': poly_coeffs,
                'fitted_values': poly_fit,
                'r_squared': 1 - np.sum((data - poly_fit)**2) / np.sum((data - np.mean(data))**2)
            }
        
        # Cumulative operations
        results['cumulative'] = {
            'cumsum': np.cumsum(data),
            'cumprod': np.cumprod(data),
            'cummax': np.maximum.accumulate(data),
            'cummin': np.minimum.accumulate(data)
        }
        
        # Rolling statistics
        window_size = min(5, len(data))
        if window_size > 1:
            rolling_mean = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data)-window_size+1)])
            
            results['rolling'] = {
                'mean': rolling_mean,
                'std': rolling_std,
                'window_size': window_size
            }
        
        # Percentile analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        results['percentiles'] = {
            f'p{p}': np.percentile(data, p) for p in percentiles
        }
        
        return results
    
    def comprehensive_analysis(self, scraped_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on all scraped data.
        """
        print("🔬 Starting comprehensive NumPy analysis...")
        
        analysis = {
            'summary': {},
            'descriptive_stats': {},
            'filtered_data': {},
            'correlations': {},
            'advanced_operations': {}
        }
        
        # Analyze each dataset
        for category, datasets in scraped_data.items():
            print(f"  📊 Analyzing {category} data...")
            
            category_stats = {}
            category_filtered = {}
            category_advanced = {}
            
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':  # numeric types
                    # Descriptive statistics
                    category_stats[data_name] = self.descriptive_statistics(data_array, f"{category}_{data_name}")
                    
                    # Filtering
                    category_filtered[data_name] = self.array_filtering(data_array, "outliers", 2.0)
                    
                    # Advanced operations
                    category_advanced[data_name] = self.advanced_array_operations(data_array)
            
            analysis['descriptive_stats'][category] = category_stats
            analysis['filtered_data'][category] = category_filtered
            analysis['advanced_operations'][category] = category_advanced
            
            # Correlation analysis within category
            numeric_data = {k: v for k, v in datasets.items() 
                           if isinstance(v, np.ndarray) and v.dtype.kind in 'biufc'}
            
            if len(numeric_data) > 1:
                analysis['correlations'][category] = self.correlation_analysis(numeric_data)
        
        # Cross-category analysis
        print("  🔗 Performing cross-category correlation analysis...")
        all_numeric_data = {}
        for category, datasets in scraped_data.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc':
                    all_numeric_data[f"{category}_{data_name}"] = data_array
        
        if len(all_numeric_data) > 1:
            analysis['correlations']['cross_category'] = self.correlation_analysis(all_numeric_data)
        
        # Generate summary
        analysis['summary'] = self._generate_summary(scraped_data, analysis)
        
        print("✅ Analysis completed!")
        return analysis
    
    def _generate_summary(self, scraped_data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate a summary of the analysis."""
        summary = {
            'total_datasets': 0,
            'total_data_points': 0,
            'categories': list(scraped_data.keys()),
            'most_variable_dataset': None,
            'most_correlated_pair': None,
            'outlier_summary': {}
        }
        
        max_cv = 0
        max_correlation = 0
        
        # Count datasets and find most variable
        for category, datasets in scraped_data.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray):
                    summary['total_datasets'] += 1
                    summary['total_data_points'] += len(data_array)
                    
                    if data_array.dtype.kind in 'biufc':  # numeric
                        # Check coefficient of variation
                        if category in analysis['descriptive_stats'] and data_name in analysis['descriptive_stats'][category]:
                            cv = analysis['descriptive_stats'][category][data_name].get('cv', 0)
                            if cv > max_cv and not np.isinf(cv):
                                max_cv = cv
                                summary['most_variable_dataset'] = f"{category}_{data_name}"
                        
                        # Check outliers
                        if category in analysis['filtered_data'] and data_name in analysis['filtered_data'][category]:
                            outlier_count = analysis['filtered_data'][category][data_name].get('outlier_count', 0)
                            summary['outlier_summary'][f"{category}_{data_name}"] = outlier_count
        
        # Find most correlated pair
        for category, corr_data in analysis['correlations'].items():
            if 'top_correlations' in corr_data and len(corr_data['top_correlations']) > 0:
                top_corr = corr_data['top_correlations'][0]
                if abs(top_corr['correlation']) > max_correlation:
                    max_correlation = abs(top_corr['correlation'])
                    summary['most_correlated_pair'] = {
                        'variables': top_corr['variables'],
                        'correlation': top_corr['correlation'],
                        'category': category
                    }
        
        return summary
    
    def create_visualizations(self, scraped_data: Dict, analysis: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualizations of the analysis results.
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Comprehensive Data Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of means across all datasets
        means = []
        labels = []
        for category, stats in analysis['descriptive_stats'].items():
            for data_name, stat_dict in stats.items():
                means.append(stat_dict['mean'])
                labels.append(f"{category}_{data_name}")
        
        if means:
            axes[0, 0].bar(range(len(means)), means)
            axes[0, 0].set_title('Mean Values Across Datasets')
            axes[0, 0].set_ylabel('Mean Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Standard deviations
        stds = []
        for category, stats in analysis['descriptive_stats'].items():
            for data_name, stat_dict in stats.items():
                stds.append(stat_dict['std'])
        
        if stds:
            axes[0, 1].bar(range(len(stds)), stds)
            axes[0, 1].set_title('Standard Deviations')
            axes[0, 1].set_ylabel('Standard Deviation')
        
        # Plot 3: Outlier counts
        outlier_counts = []
        for category, filtered in analysis['filtered_data'].items():
            for data_name, filter_dict in filtered.items():
                outlier_counts.append(filter_dict.get('outlier_count', 0))
        
        if outlier_counts:
            axes[0, 2].bar(range(len(outlier_counts)), outlier_counts)
            axes[0, 2].set_title('Outlier Counts')
            axes[0, 2].set_ylabel('Number of Outliers')
        
        # Plot 4-6: Sample data distributions
        plot_idx = 0
        for category, datasets in scraped_data.items():
            for data_name, data_array in datasets.items():
                if isinstance(data_array, np.ndarray) and data_array.dtype.kind in 'biufc' and plot_idx < 3:
                    row = 1
                    col = plot_idx
                    
                    axes[row, col].hist(data_array, bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col].set_title(f'{category} - {data_name}')
                    axes[row, col].set_ylabel('Frequency')
                    
                    plot_idx += 1
                if plot_idx >= 3:
                    break
            if plot_idx >= 3:
                break
        
        # Plot 7: Correlation heatmap
        if 'cross_category' in analysis['correlations']:
            corr_matrix = analysis['correlations']['cross_category']['correlation_matrix']
            corr_labels = analysis['correlations']['cross_category']['labels']
            
            im = axes[2, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[2, 0].set_title('Cross-Category Correlations')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[2, 0], shrink=0.8)
            cbar.set_label('Correlation')
        
        # Plot 8: Box plot of coefficient of variations
        cvs = []
        cv_labels = []
        for category, stats in analysis['descriptive_stats'].items():
            for data_name, stat_dict in stats.items():
                cv = stat_dict.get('cv', 0)
                if not np.isinf(cv):
                    cvs.append(cv)
                    cv_labels.append(f"{category}_{data_name}")
        
        if cvs:
            axes[2, 1].bar(range(len(cvs)), cvs)
            axes[2, 1].set_title('Coefficient of Variation')
            axes[2, 1].set_ylabel('CV')
            axes[2, 1].tick_params(axis='x', rotation=45)
        
        # Plot 9: Summary statistics comparison
        summary = analysis['summary']
        summary_data = [
            summary['total_datasets'],
            summary['total_data_points'] / 1000,  # Scale down for visibility
            len(summary['categories']),
            sum(summary['outlier_summary'].values())
        ]
        summary_labels = ['Datasets', 'Data Points (k)', 'Categories', 'Total Outliers']
        
        axes[2, 2].bar(summary_labels, summary_data)
        axes[2, 2].set_title('Analysis Summary')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis visualization saved to: {save_path}")
        
        return fig
    
    def print_analysis_report(self, analysis: Dict):
        """
        Print a comprehensive analysis report.
        """
        print("=" * 80)
        print("📊 COMPREHENSIVE DATA ANALYSIS REPORT")
        print("=" * 80)
        
        summary = analysis['summary']
        
        print(f"\n📈 OVERVIEW:")
        print(f"   Total datasets analyzed: {summary['total_datasets']}")
        print(f"   Total data points: {summary['total_data_points']:,}")
        print(f"   Data categories: {', '.join(summary['categories'])}")
        
        if summary['most_variable_dataset']:
            print(f"   Most variable dataset: {summary['most_variable_dataset']}")
        
        if summary['most_correlated_pair']:
            pair = summary['most_correlated_pair']
            print(f"   Strongest correlation: {pair['variables'][0]} ↔ {pair['variables'][1]}")
            print(f"                         r = {pair['correlation']:.3f}")
        
        print(f"\n🔍 DESCRIPTIVE STATISTICS SUMMARY:")
        for category, stats in analysis['descriptive_stats'].items():
            print(f"\n   {category.upper()}:")
            for data_name, stat_dict in stats.items():
                print(f"     {data_name}:")
                print(f"       Mean: {stat_dict['mean']:.2f} ± {stat_dict['std']:.2f}")
                print(f"       Range: [{stat_dict['min']:.2f}, {stat_dict['max']:.2f}]")
                print(f"       CV: {stat_dict['cv']:.3f}")
                print(f"       Skewness: {stat_dict['skewness']:.3f}")
        
        print(f"\n🎯 OUTLIER ANALYSIS:")
        total_outliers = sum(summary['outlier_summary'].values())
        print(f"   Total outliers detected: {total_outliers}")
        for dataset, count in summary['outlier_summary'].items():
            if count > 0:
                print(f"     {dataset}: {count} outliers")
        
        print(f"\n🔗 CORRELATION INSIGHTS:")
        if 'cross_category' in analysis['correlations']:
            top_corrs = analysis['correlations']['cross_category']['top_correlations'][:3]
            for i, corr in enumerate(top_corrs, 1):
                print(f"   {i}. {corr['variables'][0]} ↔ {corr['variables'][1]}")
                print(f"      Correlation: {corr['correlation']:.3f} ({corr['strength']})")
        
        print(f"\n⚡ ADVANCED ANALYSIS HIGHLIGHTS:")
        for category, advanced in analysis['advanced_operations'].items():
            print(f"\n   {category.upper()}:")
            for data_name, ops in advanced.items():
                if 'polynomial_fit' in ops:
                    r2 = ops['polynomial_fit']['r_squared']
                    print(f"     {data_name}: Polynomial fit R² = {r2:.3f}")
                
                if 'fft' in ops:
                    dom_freq_idx = ops['fft']['dominant_frequency_index']
                    print(f"     {data_name}: Dominant frequency component at index {dom_freq_idx}")
        
        print("=" * 80)