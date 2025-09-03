#!/usr/bin/env python3
"""
SIMPLIFIED VISUALIZATION GENERATOR
Autonomous Error Recovery Patterns in LLMs Research
Creates ASCII-based visualizations and data tables using only built-in libraries
"""

import json
import os
from datetime import datetime
import statistics
from typing import Dict, List

class SimpleVisualizationGenerator:
    """
    Generate text-based visualizations and formatted tables for the research paper.
    """
    
    def __init__(self):
        self.data_dir = "data"
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load the most recent data files
        self.test_results = self._load_latest_results()
        self.analysis = self._load_latest_analysis()
    
    def _load_latest_results(self) -> List[Dict]:
        """Load the most recent test results file."""
        files = [f for f in os.listdir(self.data_dir) if f.startswith("test_results_")]
        if files:
            latest = sorted(files)[-1]
            with open(os.path.join(self.data_dir, latest), 'r') as f:
                return json.load(f)
        return []
    
    def _load_latest_analysis(self) -> Dict:
        """Load the most recent analysis file."""
        files = [f for f in os.listdir(self.data_dir) if f.startswith("analysis_")]
        if files:
            latest = sorted(files)[-1]
            with open(os.path.join(self.data_dir, latest), 'r') as f:
                return json.load(f)
        return {}
    
    def generate_performance_heatmap(self):
        """Generate ASCII heatmap of performance by error type and complexity."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE HEATMAP: Success Rate by Error Type and Complexity")
        print("="*80)
        
        # Create data structure
        heatmap_data = {}
        for result in self.test_results:
            error_type = result['error_type']
            complexity = result['complexity']
            
            if error_type not in heatmap_data:
                heatmap_data[error_type] = {1: [], 2: [], 3: [], 4: [], 5: []}
            
            heatmap_data[error_type][complexity].append(1 if result['success'] else 0)
        
        # Calculate averages
        for error_type in heatmap_data:
            for complexity in heatmap_data[error_type]:
                values = heatmap_data[error_type][complexity]
                if values:
                    heatmap_data[error_type][complexity] = sum(values) / len(values) * 100
                else:
                    heatmap_data[error_type][complexity] = 0
        
        # Print heatmap
        print(f"\n{'Error Type':<25} | C1    | C2    | C3    | C4    | C5    |")
        print("-" * 70)
        
        for error_type in sorted(heatmap_data.keys()):
            row = f"{error_type:<25} |"
            for complexity in range(1, 6):
                value = heatmap_data[error_type][complexity]
                # Use different symbols for different ranges
                if value >= 80:
                    symbol = "████"  # Excellent
                elif value >= 60:
                    symbol = "███░"  # Good
                elif value >= 40:
                    symbol = "██░░"  # Fair
                elif value >= 20:
                    symbol = "█░░░"  # Poor
                else:
                    symbol = "░░░░"  # Very Poor
                row += f" {symbol} |"
            print(row)
        
        print("\nLegend: ████=80-100% ███░=60-79% ██░░=40-59% █░░░=20-39% ░░░░=0-19%")
        
        # Save to file
        output_file = os.path.join(self.viz_dir, "performance_heatmap.txt")
        with open(output_file, 'w') as f:
            f.write("PERFORMANCE HEATMAP: Success Rate by Error Type and Complexity\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Error Type':<25} | C1    | C2    | C3    | C4    | C5    |\n")
            f.write("-" * 70 + "\n")
            
            for error_type in sorted(heatmap_data.keys()):
                row = f"{error_type:<25} |"
                for complexity in range(1, 6):
                    value = heatmap_data[error_type][complexity]
                    row += f" {value:5.1f}% |"
                f.write(row + "\n")
        
        print(f"\n✓ Saved to: {output_file}")
    
    def generate_recovery_time_distribution(self):
        """Generate ASCII histogram of recovery times."""
        print("\n" + "="*80)
        print("📊 RECOVERY TIME DISTRIBUTION")
        print("="*80)
        
        # Extract recovery times
        times = [r['total_recovery_time_ms'] for r in self.test_results]
        
        if not times:
            print("No data available")
            return
        
        # Create bins
        min_time = min(times)
        max_time = max(times)
        num_bins = 10
        bin_size = (max_time - min_time) / num_bins
        
        bins = [0] * num_bins
        for time in times:
            bin_idx = min(int((time - min_time) / bin_size), num_bins - 1)
            bins[bin_idx] += 1
        
        # Find max for scaling
        max_count = max(bins)
        
        # Print histogram
        print(f"\nRecovery Time Distribution (n={len(times)} tests)")
        print("-" * 60)
        
        for i, count in enumerate(bins):
            bin_start = min_time + i * bin_size
            bin_end = bin_start + bin_size
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = "█" * bar_length
            print(f"{bin_start:6.0f}-{bin_end:6.0f}ms | {bar:<40} | {count:3d}")
        
        # Statistics
        print("\n" + "-" * 60)
        print(f"Mean: {statistics.mean(times):.2f}ms")
        print(f"Median: {statistics.median(times):.2f}ms")
        print(f"Std Dev: {statistics.stdev(times):.2f}ms" if len(times) > 1 else "Std Dev: N/A")
        print(f"Min: {min(times):.2f}ms")
        print(f"Max: {max(times):.2f}ms")
        
        # Save to file
        output_file = os.path.join(self.viz_dir, "recovery_time_distribution.txt")
        with open(output_file, 'w') as f:
            f.write("RECOVERY TIME DISTRIBUTION\n")
            f.write("="*60 + "\n\n")
            for i, count in enumerate(bins):
                bin_start = min_time + i * bin_size
                bin_end = bin_start + bin_size
                f.write(f"{bin_start:6.0f}-{bin_end:6.0f}ms: {count} tests\n")
            f.write("\n" + "-"*60 + "\n")
            f.write(f"Mean: {statistics.mean(times):.2f}ms\n")
            f.write(f"Median: {statistics.median(times):.2f}ms\n")
            if len(times) > 1:
                f.write(f"Std Dev: {statistics.stdev(times):.2f}ms\n")
            f.write(f"Min: {min(times):.2f}ms\n")
            f.write(f"Max: {max(times):.2f}ms\n")
        
        print(f"\n✓ Saved to: {output_file}")
    
    def generate_performance_by_complexity(self):
        """Generate bar chart of performance by complexity level."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE BY COMPLEXITY LEVEL")
        print("="*80)
        
        # Aggregate by complexity
        complexity_data = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for result in self.test_results:
            complexity = result['complexity']
            complexity_data[complexity].append(1 if result['success'] else 0)
        
        # Calculate success rates
        print("\nSuccess Rate by Complexity Level:")
        print("-" * 50)
        
        for complexity in range(1, 6):
            values = complexity_data[complexity]
            if values:
                success_rate = sum(values) / len(values) * 100
                total_tests = len(values)
                
                # Create bar
                bar_length = int(success_rate / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                
                print(f"Level {complexity}: {bar:<50} {success_rate:6.2f}% ({total_tests} tests)")
        
        # Save detailed table
        output_file = os.path.join(self.viz_dir, "performance_by_complexity.txt")
        with open(output_file, 'w') as f:
            f.write("PERFORMANCE BY COMPLEXITY LEVEL\n")
            f.write("="*60 + "\n\n")
            f.write("Complexity | Success Rate | Total Tests | Successful | Failed\n")
            f.write("-"*60 + "\n")
            
            for complexity in range(1, 6):
                values = complexity_data[complexity]
                if values:
                    success_rate = sum(values) / len(values) * 100
                    successful = sum(values)
                    failed = len(values) - successful
                    f.write(f"Level {complexity}    | {success_rate:11.2f}% | {len(values):11d} | {successful:10d} | {failed:6d}\n")
        
        print(f"\n✓ Saved to: {output_file}")
    
    def generate_error_type_ranking(self):
        """Generate ranking table of error types by recovery success."""
        print("\n" + "="*80)
        print("📊 ERROR TYPE RANKING BY RECOVERY SUCCESS")
        print("="*80)
        
        # Aggregate by error type
        error_data = {}
        
        for result in self.test_results:
            error_type = result['error_type']
            if error_type not in error_data:
                error_data[error_type] = {
                    'success': 0,
                    'total': 0,
                    'times': []
                }
            
            error_data[error_type]['total'] += 1
            if result['success']:
                error_data[error_type]['success'] += 1
            error_data[error_type]['times'].append(result['total_recovery_time_ms'])
        
        # Calculate metrics and sort
        rankings = []
        for error_type, data in error_data.items():
            success_rate = (data['success'] / data['total']) * 100 if data['total'] > 0 else 0
            avg_time = statistics.mean(data['times']) if data['times'] else 0
            
            rankings.append({
                'error_type': error_type,
                'success_rate': success_rate,
                'avg_time': avg_time,
                'total_tests': data['total'],
                'successful': data['success']
            })
        
        # Sort by success rate
        rankings.sort(key=lambda x: x['success_rate'], reverse=True)
        
        # Print ranking table
        print("\nRank | Error Type                    | Success | Avg Time | Tests")
        print("-" * 70)
        
        for i, item in enumerate(rankings, 1):
            print(f"{i:4d} | {item['error_type']:<29} | {item['success_rate']:6.1f}% | {item['avg_time']:7.1f}ms | {item['total_tests']:5d}")
        
        # Save to file
        output_file = os.path.join(self.viz_dir, "error_type_ranking.txt")
        with open(output_file, 'w') as f:
            f.write("ERROR TYPE RANKING BY RECOVERY SUCCESS\n")
            f.write("="*70 + "\n\n")
            f.write("Rank | Error Type                    | Success | Avg Time | Tests | Successful\n")
            f.write("-"*80 + "\n")
            
            for i, item in enumerate(rankings, 1):
                f.write(f"{i:4d} | {item['error_type']:<29} | {item['success_rate']:6.1f}% | "
                       f"{item['avg_time']:7.1f}ms | {item['total_tests']:5d} | {item['successful']:10d}\n")
        
        print(f"\n✓ Saved to: {output_file}")
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix between key metrics."""
        print("\n" + "="*80)
        print("📊 CORRELATION MATRIX")
        print("="*80)
        
        # Extract metrics
        complexities = [r['complexity'] for r in self.test_results]
        times = [r['total_recovery_time_ms'] for r in self.test_results]
        confidences = [r['confidence_score'] for r in self.test_results]
        qualities = [r['explanation_quality'] for r in self.test_results]
        
        # Simple correlation calculation
        def correlation(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0
            
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            
            cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
            std_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
            std_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
            
            if std_x == 0 or std_y == 0:
                return 0
            
            return cov / (std_x * std_y)
        
        # Calculate correlations
        metrics = {
            'Complexity': complexities,
            'Recovery Time': times,
            'Confidence': confidences,
            'Quality': qualities
        }
        
        print("\nCorrelation Matrix:")
        print("-" * 60)
        print(f"{'Metric':<15} | {'Complexity':<10} | {'Time':<10} | {'Confidence':<10} | {'Quality':<10}")
        print("-" * 60)
        
        for metric1_name, metric1_data in metrics.items():
            row = f"{metric1_name:<15} |"
            for metric2_name, metric2_data in metrics.items():
                corr = correlation(metric1_data, metric2_data)
                row += f" {corr:9.3f} |"
            print(row)
        
        # Key insights
        print("\n" + "-" * 60)
        print("Key Correlations:")
        print(f"  • Complexity vs Recovery Time: {correlation(complexities, times):.3f}")
        print(f"  • Complexity vs Confidence: {correlation(complexities, confidences):.3f}")
        print(f"  • Recovery Time vs Confidence: {correlation(times, confidences):.3f}")
        print(f"  • Confidence vs Quality: {correlation(confidences, qualities):.3f}")
        
        # Save to file
        output_file = os.path.join(self.viz_dir, "correlation_matrix.txt")
        with open(output_file, 'w') as f:
            f.write("CORRELATION MATRIX\n")
            f.write("="*60 + "\n\n")
            f.write("Correlations between key experimental metrics\n\n")
            
            f.write(f"{'Metric':<15} | {'Complexity':<10} | {'Time':<10} | {'Confidence':<10} | {'Quality':<10}\n")
            f.write("-"*60 + "\n")
            
            for metric1_name, metric1_data in metrics.items():
                row = f"{metric1_name:<15} |"
                for metric2_name, metric2_data in metrics.items():
                    corr = correlation(metric1_data, metric2_data)
                    row += f" {corr:9.3f} |"
                f.write(row + "\n")
        
        print(f"\n✓ Saved to: {output_file}")
    
    def generate_summary_dashboard(self):
        """Generate comprehensive summary dashboard."""
        print("\n" + "="*80)
        print("📊 EXPERIMENTAL SUMMARY DASHBOARD")
        print("="*80)
        
        # Calculate key metrics
        total_tests = len(self.test_results)
        successful = sum(1 for r in self.test_results if r['success'])
        detected = sum(1 for r in self.test_results if r.get('error_detected', False))
        corrected = sum(1 for r in self.test_results if r.get('correction_valid', False))
        
        times = [r['total_recovery_time_ms'] for r in self.test_results]
        
        dashboard = f"""
╔════════════════════════════════════════════════════════════════════════╗
║           AUTONOMOUS ERROR RECOVERY PATTERNS IN LLMs                      ║
║                    EXPERIMENTAL RESULTS DASHBOARD                         ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  OVERALL METRICS                                                          ║
║  ├─ Total Tests Executed: {total_tests:<47}║
║  ├─ Successful Recoveries: {successful} ({(successful/total_tests)*100:.1f}%)                                    ║
║  ├─ Error Detection Rate: {detected} ({(detected/total_tests)*100:.1f}%)                                     ║
║  └─ Correction Success: {corrected} ({(corrected/detected)*100 if detected > 0 else 0:.1f}%)                                       ║
║                                                                            ║
║  TIMING STATISTICS (milliseconds)                                         ║
║  ├─ Mean Recovery Time: {statistics.mean(times):.2f}                                          ║
║  ├─ Median Recovery Time: {statistics.median(times):.2f}                                        ║
║  ├─ Minimum Time: {min(times):.2f}                                                ║
║  └─ Maximum Time: {max(times):.2f}                                                ║
║                                                                            ║
║  COMPLEXITY ANALYSIS                                                      ║
║  ├─ Level 1 Success Rate: {self._get_complexity_rate(1):.1f}%                                      ║
║  ├─ Level 2 Success Rate: {self._get_complexity_rate(2):.1f}%                                      ║
║  ├─ Level 3 Success Rate: {self._get_complexity_rate(3):.1f}%                                      ║
║  ├─ Level 4 Success Rate: {self._get_complexity_rate(4):.1f}%                                      ║
║  └─ Level 5 Success Rate: {self._get_complexity_rate(5):.1f}%                                      ║
║                                                                            ║
║  TOP PERFORMING ERROR TYPES                                               ║
"""
        
        # Get top 3 error types
        error_rates = {}
        for result in self.test_results:
            error_type = result['error_type']
            if error_type not in error_rates:
                error_rates[error_type] = []
            error_rates[error_type].append(1 if result['success'] else 0)
        
        error_performance = []
        for error_type, values in error_rates.items():
            if values:
                rate = sum(values) / len(values) * 100
                error_performance.append((error_type, rate))
        
        error_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (error_type, rate) in enumerate(error_performance[:3], 1):
            dashboard += f"║  {i}. {error_type:<25} ({rate:.1f}%)                             ║\n"
        
        dashboard += """║                                                                            ║
╚════════════════════════════════════════════════════════════════════════╝
"""
        
        print(dashboard)
        
        # Save to file
        output_file = os.path.join(self.viz_dir, "summary_dashboard.txt")
        with open(output_file, 'w') as f:
            f.write(dashboard)
        
        print(f"✓ Saved to: {output_file}")
    
    def _get_complexity_rate(self, complexity: int) -> float:
        """Helper to get success rate for a complexity level."""
        tests = [r for r in self.test_results if r['complexity'] == complexity]
        if tests:
            successful = sum(1 for r in tests if r['success'])
            return (successful / len(tests)) * 100
        return 0.0
    
    def generate_all_visualizations(self):
        """Generate all visualizations and tables."""
        print("\n" + "🎨" * 50)
        print("\nGENERATING ALL VISUALIZATIONS AND TABLES")
        print("=" * 100)
        
        # Generate each visualization
        self.generate_performance_heatmap()
        self.generate_recovery_time_distribution()
        self.generate_performance_by_complexity()
        self.generate_error_type_ranking()
        self.generate_correlation_matrix()
        self.generate_summary_dashboard()
        
        print("\n" + "="*100)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print(f"📁 Output directory: {self.viz_dir}")
        print("="*100)
        
        # List all generated files
        print("\nGenerated Files:")
        for file in os.listdir(self.viz_dir):
            if file.endswith('.txt'):
                print(f"  • {file}")
        
        return True

def main():
    """Main execution function."""
    print("\n" + "🎨" * 50)
    print("\n     VISUALIZATION GENERATOR FOR AUTONOMOUS ERROR RECOVERY RESEARCH")
    print("\n" + "🎨" * 50)
    
    try:
        # Initialize generator
        generator = SimpleVisualizationGenerator()
        
        # Check if we have data
        if not generator.test_results:
            print("\n⚠️  No test results found. Please run the experiment first.")
            return False
        
        print(f"\n✓ Loaded {len(generator.test_results)} test results")
        print("📊 Starting visualization generation...\n")
        
        # Generate all visualizations
        success = generator.generate_all_visualizations()
        
        if success:
            print("\n🎉 Visualization generation complete!")
            print("📊 All tables and charts have been created")
            print("📁 Check the 'visualizations' folder for output files")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: Visualization generation failed!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)