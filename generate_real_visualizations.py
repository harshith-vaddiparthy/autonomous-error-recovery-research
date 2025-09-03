#!/usr/bin/env python3
"""
PROFESSIONAL VISUALIZATION GENERATOR
Autonomous Error Recovery Patterns in LLMs Research
Generates publication-quality graphs, charts, and visualizations
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# IEEE paper column width in inches
IEEE_COLUMN_WIDTH = 3.5
IEEE_PAGE_WIDTH = 7.16

class ProfessionalVisualizationGenerator:
    """Generate publication-quality visualizations for the research paper."""
    
    def __init__(self):
        self.data_dir = "data"
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set color palette for consistency
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#73BA9B',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'dark': '#003D5B',
            'light': '#F0F3BD'
        }
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Load data
        self.test_results = self._load_latest_results()
        self.df = pd.DataFrame(self.test_results)
        
        print(f"✓ Loaded {len(self.test_results)} test results")
        print(f"✓ Data shape: {self.df.shape}")
    
    def _load_latest_results(self):
        """Load the most recent test results file."""
        files = [f for f in os.listdir(self.data_dir) if f.startswith("test_results_")]
        if files:
            latest = sorted(files)[-1]
            with open(os.path.join(self.data_dir, latest), 'r') as f:
                return json.load(f)
        return []
    
    def generate_performance_heatmap(self):
        """Generate heatmap showing performance by error type and complexity."""
        print("\n📊 Generating Performance Heatmap...")
        
        # Prepare data
        pivot_data = self.df.pivot_table(
            values='success',
            index='error_type',
            columns='complexity',
            aggfunc=lambda x: sum(x) / len(x) * 100
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 6))
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Success Rate (%)'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title('Error Recovery Success Rate by Type and Complexity', fontsize=14, fontweight='bold')
        plt.xlabel('Complexity Level', fontsize=12)
        plt.ylabel('Error Type', fontsize=12)
        
        # Rotate y-axis labels for readability
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'performance_heatmap.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_recovery_time_distribution(self):
        """Generate histogram and box plot of recovery times."""
        print("\n📊 Generating Recovery Time Distribution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, 3.5))
        
        # Histogram
        ax1.hist(
            self.df['total_recovery_time_ms'],
            bins=20,
            color=self.colors['primary'],
            edgecolor='black',
            alpha=0.7
        )
        ax1.axvline(
            self.df['total_recovery_time_ms'].mean(),
            color=self.colors['danger'],
            linestyle='--',
            linewidth=2,
            label=f'Mean: {self.df["total_recovery_time_ms"].mean():.1f}ms'
        )
        ax1.axvline(
            self.df['total_recovery_time_ms'].median(),
            color=self.colors['success'],
            linestyle=':',
            linewidth=2,
            label=f'Median: {self.df["total_recovery_time_ms"].median():.1f}ms'
        )
        
        ax1.set_xlabel('Recovery Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Recovery Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by complexity
        complexity_data = [
            self.df[self.df['complexity'] == i]['total_recovery_time_ms'].values
            for i in range(1, 6)
        ]
        
        bp = ax2.boxplot(
            complexity_data,
            labels=[f'L{i}' for i in range(1, 6)],
            patch_artist=True
        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], sns.color_palette("coolwarm", 5)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Complexity Level')
        ax2.set_ylabel('Recovery Time (ms)')
        ax2.set_title('Recovery Time by Complexity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'recovery_time_distribution.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_performance_by_complexity(self):
        """Generate bar chart showing performance metrics by complexity level."""
        print("\n📊 Generating Performance by Complexity...")
        
        # Aggregate data
        complexity_stats = self.df.groupby('complexity').agg({
            'success': lambda x: sum(x) / len(x) * 100,
            'total_recovery_time_ms': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(IEEE_PAGE_WIDTH, 3))
        
        # Success Rate
        axes[0].bar(
            complexity_stats['complexity'],
            complexity_stats['success'],
            color=sns.color_palette("viridis", 5),
            edgecolor='black',
            linewidth=1
        )
        axes[0].set_xlabel('Complexity Level')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(complexity_stats['success']):
            axes[0].text(i + 1, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
        
        # Recovery Time
        axes[1].plot(
            complexity_stats['complexity'],
            complexity_stats['total_recovery_time_ms'],
            marker='o',
            markersize=8,
            linewidth=2,
            color=self.colors['danger']
        )
        axes[1].fill_between(
            complexity_stats['complexity'],
            complexity_stats['total_recovery_time_ms'],
            alpha=0.3,
            color=self.colors['danger']
        )
        axes[1].set_xlabel('Complexity Level')
        axes[1].set_ylabel('Avg Recovery Time (ms)')
        axes[1].set_title('Recovery Time')
        axes[1].grid(True, alpha=0.3)
        
        # Confidence Score
        axes[2].bar(
            complexity_stats['complexity'],
            complexity_stats['confidence_score'],
            color=sns.color_palette("muted", 5),
            edgecolor='black',
            linewidth=1
        )
        axes[2].set_xlabel('Complexity Level')
        axes[2].set_ylabel('Confidence Score')
        axes[2].set_title('Confidence Score')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Performance Metrics by Complexity Level', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'performance_by_complexity.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_error_type_ranking(self):
        """Generate horizontal bar chart of error types ranked by success rate."""
        print("\n📊 Generating Error Type Ranking...")
        
        # Aggregate by error type
        error_stats = self.df.groupby('error_type').agg({
            'success': lambda x: sum(x) / len(x) * 100,
            'total_recovery_time_ms': 'mean'
        }).reset_index()
        
        # Sort by success rate
        error_stats = error_stats.sort_values('success', ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 5))
        
        # Create horizontal bar chart
        bars = ax.barh(
            range(len(error_stats)),
            error_stats['success'],
            color=[self.colors['success'] if x >= 60 else 
                   self.colors['warning'] if x >= 40 else 
                   self.colors['danger'] for x in error_stats['success']],
            edgecolor='black',
            linewidth=1
        )
        
        # Customize
        ax.set_yticks(range(len(error_stats)))
        ax.set_yticklabels(error_stats['error_type'])
        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_title('Error Recovery Success Rate by Error Type', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, error_stats['success'])):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}%', va='center', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'error_type_ranking.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix heatmap."""
        print("\n📊 Generating Correlation Matrix...")
        
        # Select numeric columns for correlation
        numeric_cols = ['complexity', 'total_recovery_time_ms', 'confidence_score', 'explanation_quality']
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH * 1.5, IEEE_COLUMN_WIDTH * 1.5))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        # Customize labels
        ax.set_xticklabels(['Complexity', 'Recovery Time', 'Confidence', 'Quality'], rotation=45, ha='right')
        ax.set_yticklabels(['Complexity', 'Recovery Time', 'Confidence', 'Quality'], rotation=0)
        
        plt.title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'correlation_matrix.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_pie_charts(self):
        """Generate pie charts for overall performance breakdown."""
        print("\n📊 Generating Pie Charts...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, 3.5))
        
        # Overall Success Distribution
        success_counts = self.df['success'].value_counts()
        labels1 = ['Successful', 'Failed']
        sizes1 = [success_counts.get(True, 0), success_counts.get(False, 0)]
        colors1 = [self.colors['success'], self.colors['danger']]
        
        wedges1, texts1, autotexts1 = ax1.pie(
            sizes1,
            labels=labels1,
            colors=colors1,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0)
        )
        
        ax1.set_title('Overall Recovery Success Rate')
        
        # Error Category Distribution
        error_categories = {
            'Syntactic': ['missing_parenthesis', 'unclosed_quotes', 'invalid_indentation', 
                         'missing_colon', 'bracket_mismatch'],
            'Runtime': ['division_by_zero', 'none_type_access', 'index_out_of_bounds', 'key_error'],
            'Logic': ['infinite_loop', 'off_by_one_error', 'incorrect_condition', 'missing_base_case'],
            'Semantic': ['string_int_concatenation', 'incorrect_api_usage']
        }
        
        category_success = {}
        for category, errors in error_categories.items():
            category_df = self.df[self.df['error_type'].isin(errors)]
            if len(category_df) > 0:
                category_success[category] = (category_df['success'].sum() / len(category_df)) * 100
        
        labels2 = list(category_success.keys())
        sizes2 = list(category_success.values())
        colors2 = sns.color_palette("Set2", len(labels2))
        
        wedges2, texts2, autotexts2 = ax2.pie(
            sizes2,
            labels=labels2,
            colors=colors2,
            autopct='%1.1f%%',
            startangle=45
        )
        
        ax2.set_title('Success Rate by Error Category')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'pie_charts.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_violin_plots(self):
        """Generate violin plots for recovery time distribution."""
        print("\n📊 Generating Violin Plots...")
        
        fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 4))
        
        # Create violin plot
        parts = ax.violinplot(
            [self.df[self.df['complexity'] == i]['total_recovery_time_ms'].values 
             for i in range(1, 6)],
            positions=range(1, 6),
            widths=0.7,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor(self.colors['primary'])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Customize other elements
        parts['cmeans'].set_color(self.colors['danger'])
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color(self.colors['success'])
        parts['cmedians'].set_linewidth(2)
        
        ax.set_xlabel('Complexity Level', fontsize=12)
        ax.set_ylabel('Recovery Time (ms)', fontsize=12)
        ax.set_title('Recovery Time Distribution by Complexity (Violin Plot)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f'Level {i}' for i in range(1, 6)])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['danger'], linewidth=2, label='Mean'),
            Line2D([0], [0], color=self.colors['success'], linewidth=2, label='Median')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'violin_plots.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_3d_surface_plot(self):
        """Generate 3D surface plot showing relationship between variables."""
        print("\n📊 Generating 3D Surface Plot...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(IEEE_PAGE_WIDTH, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        complexity_range = np.arange(1, 6)
        error_types = self.df['error_type'].unique()
        error_type_indices = np.arange(len(error_types))
        
        X, Y = np.meshgrid(complexity_range, error_type_indices)
        Z = np.zeros_like(X, dtype=float)
        
        for i, error_type in enumerate(error_types):
            for j, complexity in enumerate(complexity_range):
                subset = self.df[(self.df['error_type'] == error_type) & 
                               (self.df['complexity'] == complexity)]
                if len(subset) > 0:
                    Z[i, j] = (subset['success'].sum() / len(subset)) * 100
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              edgecolor='none', antialiased=True)
        
        # Customize axes
        ax.set_xlabel('Complexity Level', fontsize=10)
        ax.set_ylabel('Error Type Index', fontsize=10)
        ax.set_zlabel('Success Rate (%)', fontsize=10)
        ax.set_title('3D Success Rate Surface', fontsize=12, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, '3d_surface_plot.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_summary_dashboard(self):
        """Generate comprehensive dashboard with multiple panels."""
        print("\n📊 Generating Summary Dashboard...")
        
        fig = plt.figure(figsize=(IEEE_PAGE_WIDTH * 2, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Overall Metrics
        ax1 = fig.add_subplot(gs[0, :])
        metrics_text = f"""
        EXPERIMENTAL RESULTS SUMMARY
        
        Total Tests: {len(self.df)}
        Success Rate: {(self.df['success'].sum() / len(self.df)) * 100:.1f}%
        Mean Recovery Time: {self.df['total_recovery_time_ms'].mean():.1f}ms
        Median Recovery Time: {self.df['total_recovery_time_ms'].median():.1f}ms
        Std Dev: {self.df['total_recovery_time_ms'].std():.1f}ms
        """
        ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light']))
        ax1.axis('off')
        
        # Panel 2: Success by Complexity
        ax2 = fig.add_subplot(gs[1, 0])
        complexity_success = self.df.groupby('complexity')['success'].apply(
            lambda x: sum(x) / len(x) * 100
        )
        ax2.bar(complexity_success.index, complexity_success.values,
               color=sns.color_palette("coolwarm", 5))
        ax2.set_xlabel('Complexity')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success by Complexity')
        ax2.set_ylim(0, 100)
        
        # Panel 3: Top 5 Error Types
        ax3 = fig.add_subplot(gs[1, 1])
        error_success = self.df.groupby('error_type')['success'].apply(
            lambda x: sum(x) / len(x) * 100
        ).sort_values(ascending=False).head(5)
        ax3.barh(range(5), error_success.values, 
                color=self.colors['success'])
        ax3.set_yticks(range(5))
        ax3.set_yticklabels(error_success.index, fontsize=8)
        ax3.set_xlabel('Success Rate (%)')
        ax3.set_title('Top 5 Error Types')
        
        # Panel 4: Time Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(self.df['total_recovery_time_ms'], bins=15,
                color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax4.axvline(self.df['total_recovery_time_ms'].mean(), 
                   color='red', linestyle='--', label='Mean')
        ax4.set_xlabel('Recovery Time (ms)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Time Distribution')
        ax4.legend()
        
        # Panel 5: Correlation Heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        numeric_cols = ['complexity', 'total_recovery_time_ms', 'confidence_score']
        corr = self.df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax5,
                   cbar_kws={'label': 'Correlation'})
        ax5.set_title('Correlation Matrix')
        
        # Panel 6: Confidence Score Distribution
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.boxplot([self.df[self.df['complexity'] == i]['confidence_score'].values 
                    for i in range(1, 6)],
                   labels=[f'L{i}' for i in range(1, 6)])
        ax6.set_xlabel('Complexity')
        ax6.set_ylabel('Confidence Score')
        ax6.set_title('Confidence Distribution')
        
        plt.suptitle('Autonomous Error Recovery Patterns - Experimental Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        output_path = os.path.join(self.viz_dir, 'summary_dashboard.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*80)
        print("🎨 GENERATING PROFESSIONAL VISUALIZATIONS")
        print("="*80)
        
        try:
            self.generate_performance_heatmap()
            self.generate_recovery_time_distribution()
            self.generate_performance_by_complexity()
            self.generate_error_type_ranking()
            self.generate_correlation_matrix()
            self.generate_pie_charts()
            self.generate_violin_plots()
            self.generate_3d_surface_plot()
            self.generate_summary_dashboard()
            
            print("\n" + "="*80)
            print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
            print(f"📁 Output directory: {self.viz_dir}")
            print("="*80)
            
            # List generated files
            print("\n📊 Generated Files:")
            for file in os.listdir(self.viz_dir):
                if file.endswith(('.pdf', '.png')):
                    file_path = os.path.join(self.viz_dir, file)
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  • {file} ({size:.1f} KB)")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function."""
    print("\n" + "🎨" * 50)
    print("\n   PROFESSIONAL VISUALIZATION GENERATOR FOR AUTONOMOUS ERROR RECOVERY RESEARCH")
    print("\n" + "🎨" * 50)
    
    try:
        generator = ProfessionalVisualizationGenerator()
        
        if not generator.test_results:
            print("\n⚠️  No test results found. Please run the experiment first.")
            return False
        
        print(f"\n📊 Starting professional visualization generation...")
        print(f"📈 Creating publication-quality graphs and charts...")
        
        success = generator.generate_all_visualizations()
        
        if success:
            print("\n🎉 VISUALIZATION GENERATION COMPLETE!")
            print("📊 All professional graphs, charts, and visualizations have been created")
            print("📁 Check the 'visualizations' folder for high-quality PDF and PNG files")
            print("✨ Ready for inclusion in IEEE paper")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)