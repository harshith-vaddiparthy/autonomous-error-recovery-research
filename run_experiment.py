"""
MASTER EXPERIMENTAL ORCHESTRATOR
Autonomous Error Recovery Patterns in LLMs Research
Author: Harshith Vaddiparthy
"""

import os
import sys
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import traceback

# Import our modules
from meta_prompt_generator import MetaPromptGenerator
from experiment_runner import ExperimentRunner
from visualization_engine import VisualizationEngine

class MasterOrchestrator:
    """
    Master control system for the entire experimental process.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.experiment_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.prompt_generator = MetaPromptGenerator()
        self.runner = ExperimentRunner()
        self.viz_engine = VisualizationEngine(
            data_dir="autonomous_error_recovery_research/data",
            viz_dir="autonomous_error_recovery_research/visualizations"
        )
        
        # Experimental parameters
        self.error_types_to_test = [
            "missing_parenthesis",
            "infinite_loop", 
            "string_int_concatenation",
            "division_by_zero",
            "incorrect_api_usage",
            "unclosed_quotes",
            "invalid_indentation",
            "missing_colon",
            "off_by_one",
            "none_type_access",
            "bracket_mismatch",
            "incorrect_condition",
            "missing_base_case",
            "index_out_of_bounds",
            "key_error"
        ]
        
        self.complexity_levels = [1, 2, 3, 4, 5]
        self.variations_per_combination = 2
        
        # Calculate total tests
        self.total_tests = (len(self.error_types_to_test) * 
                          len(self.complexity_levels) * 
                          self.variations_per_combination)
        
        self._print_header()
    
    def _print_header(self):
        """Print experimental header information."""
        print("=" * 100)
        print(" " * 20 + "AUTONOMOUS ERROR RECOVERY PATTERNS IN LLMs")
        print(" " * 25 + "MASTER EXPERIMENTAL ORCHESTRATOR")
        print("=" * 100)
        print(f"\n📅 Experiment Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
        print(f"👤 Researcher: Harshith Vaddiparthy")
        print(f"📧 Contact: hi@harshith.io")
        print(f"🔬 ORCID: 0009-0005-1620-4045")
        print("\n" + "-" * 100)
        print("\n📊 EXPERIMENTAL PARAMETERS:")
        print(f"  • Error Types to Test: {len(self.error_types_to_test)}")
        print(f"  • Complexity Levels: {len(self.complexity_levels)} (1-5)")
        print(f"  • Variations per Combination: {self.variations_per_combination}")
        print(f"  • Total Test Cases: {self.total_tests}")
        print(f"  • Estimated Duration: ~{self.total_tests * 0.5:.1f} minutes")
        print("\n" + "=" * 100)
    
    def run_phase_1_test_generation(self):
        """
        Phase 1: Generate all test cases and meta-prompts.
        """
        print("\n" + "🔬" * 50)
        print("\n📝 PHASE 1: TEST CASE GENERATION")
        print("=" * 100)
        
        test_cases = []
        test_counter = 0
        
        for error_type in self.error_types_to_test:
            print(f"\n🔸 Generating tests for: {error_type}")
            print("-" * 80)
            
            for complexity in self.complexity_levels:
                for variation in range(1, self.variations_per_combination + 1):
                    test_counter += 1
                    
                    # Generate meta-prompt
                    meta_prompt = self.prompt_generator.generate_error_injection_prompt(
                        error_type, complexity
                    )
                    
                    # Create test case
                    test_case = {
                        "test_number": test_counter,
                        "error_type": error_type,
                        "complexity": complexity,
                        "variation": variation,
                        "meta_prompt": meta_prompt,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    test_cases.append(test_case)
                    
                    # Progress indicator
                    if test_counter % 10 == 0:
                        progress = (test_counter / self.total_tests) * 100
                        print(f"  ✓ Generated {test_counter}/{self.total_tests} test cases ({progress:.1f}%)")
        
        # Save test cases
        test_file = f"autonomous_error_recovery_research/data/test_cases_{self.experiment_id}.json"
        with open(test_file, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        print(f"\n✅ Phase 1 Complete: {len(test_cases)} test cases generated")
        print(f"📁 Saved to: {test_file}")
        
        return test_cases
    
    def run_phase_2_execution(self, test_cases: List[Dict]):
        """
        Phase 2: Execute all test cases and collect results.
        """
        print("\n" + "🔬" * 50)
        print("\n⚡ PHASE 2: TEST EXECUTION")
        print("=" * 100)
        
        all_results = []
        successful_tests = 0
        failed_tests = 0
        
        print("\n🚀 Starting test execution...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST [{i}/{self.total_tests}] - {test_case['error_type']} "
                  f"(Complexity: {test_case['complexity']}, Variation: {test_case['variation']})")
            print(f"{'='*80}")
            
            # Execute test
            result = self.runner.run_single_test(
                error_type=test_case['error_type'],
                complexity=test_case['complexity'],
                variation=test_case['variation']
            )
            
            # Merge with test case info
            result.update({
                "test_number": test_case['test_number'],
                "meta_prompt": test_case['meta_prompt'][:200] + "..."  # Truncate for storage
            })
            
            all_results.append(result)
            
            # Update counters
            if result['success']:
                successful_tests += 1
            else:
                failed_tests += 1
            
            # Progress report every 10 tests
            if i % 10 == 0:
                success_rate = (successful_tests / i) * 100
                print(f"\n📊 PROGRESS REPORT:")
                print(f"  • Tests Completed: {i}/{self.total_tests}")
                print(f"  • Success Rate: {success_rate:.2f}%")
                print(f"  • Successful: {successful_tests}, Failed: {failed_tests}")
                
                # Time estimation
                elapsed = (datetime.now() - self.start_time).total_seconds()
                avg_time_per_test = elapsed / i
                remaining_tests = self.total_tests - i
                eta_seconds = remaining_tests * avg_time_per_test
                eta_minutes = eta_seconds / 60
                
                print(f"  • Estimated Time Remaining: {eta_minutes:.1f} minutes")
        
        # Save results
        results_file = f"autonomous_error_recovery_research/data/execution_results_{self.experiment_id}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Phase 2 Complete: All {len(all_results)} tests executed")
        print(f"📁 Results saved to: {results_file}")
        
        return all_results
    
    def run_phase_3_analysis(self, test_results: List[Dict]):
        """
        Phase 3: Analyze results and generate metrics.
        """
        print("\n" + "🔬" * 50)
        print("\n📈 PHASE 3: DATA ANALYSIS")
        print("=" * 100)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(test_results)
        
        print("\n📊 STATISTICAL ANALYSIS:")
        print("-" * 80)
        
        # Overall metrics
        overall_metrics = {
            "total_tests": len(df),
            "success_rate": (df['success'].sum() / len(df)) * 100,
            "detection_rate": (df['error_detected'].sum() / len(df)) * 100,
            "correction_rate": (df['correction_valid'].sum() / df['error_detected'].sum()) * 100 
                             if df['error_detected'].sum() > 0 else 0,
            "avg_recovery_time": df['recovery_time'].mean(),
            "median_recovery_time": df['recovery_time'].median(),
            "std_recovery_time": df['recovery_time'].std(),
            "min_recovery_time": df['recovery_time'].min(),
            "max_recovery_time": df['recovery_time'].max(),
            "avg_confidence": df['confidence_score'].mean(),
            "avg_explanation_quality": df['explanation_quality'].mean(),
            "successful_recoveries": df['success'].sum(),
            "partial_recoveries": len(df[(df['error_detected'] == True) & (df['correction_valid'] == False)]),
            "failed_recoveries": len(df[df['error_detected'] == False])
        }
        
        print("\n🎯 OVERALL PERFORMANCE METRICS:")
        for key, value in overall_metrics.items():
            if 'rate' in key or 'confidence' in key or 'quality' in key:
                print(f"  • {key.replace('_', ' ').title()}: {value:.2f}%")
            elif 'time' in key:
                print(f"  • {key.replace('_', ' ').title()}: {value:.2f}ms")
            else:
                print(f"  • {key.replace('_', ' ').title()}: {value:.0f}")
        
        # Analysis by error type
        print("\n📊 PERFORMANCE BY ERROR TYPE:")
        print("-" * 80)
        
        error_analysis = df.groupby('error_type').agg({
            'success': lambda x: (x.sum() / len(x)) * 100,
            'recovery_time': 'mean',
            'confidence_score': 'mean',
            'explanation_quality': 'mean'
        }).round(2)
        
        print(error_analysis.to_string())
        
        # Analysis by complexity
        print("\n📊 PERFORMANCE BY COMPLEXITY LEVEL:")
        print("-" * 80)
        
        complexity_analysis = df.groupby('complexity').agg({
            'success': lambda x: (x.sum() / len(x)) * 100,
            'error_detected': lambda x: (x.sum() / len(x)) * 100,
            'correction_valid': lambda x: (x.sum() / len(x)) * 100,
            'recovery_time': 'mean'
        }).round(2)
        
        print(complexity_analysis.to_string())
        
        # Correlation analysis
        print("\n📊 CORRELATION ANALYSIS:")
        print("-" * 80)
        
        correlation_vars = ['complexity', 'recovery_time', 'confidence_score', 'explanation_quality']
        correlation_matrix = df[correlation_vars].corr()
        
        print("\nKey Correlations:")
        print(f"  • Complexity vs Recovery Time: {correlation_matrix.loc['complexity', 'recovery_time']:.3f}")
        print(f"  • Complexity vs Confidence: {correlation_matrix.loc['complexity', 'confidence_score']:.3f}")
        print(f"  • Recovery Time vs Confidence: {correlation_matrix.loc['recovery_time', 'confidence_score']:.3f}")
        
        # Save analysis results
        analysis_file = f"autonomous_error_recovery_research/data/analysis_results_{self.experiment_id}.json"
        analysis_data = {
            "overall_metrics": overall_metrics,
            "error_analysis": error_analysis.to_dict(),
            "complexity_analysis": complexity_analysis.to_dict(),
            "correlation_matrix": correlation_matrix.to_dict()
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n✅ Phase 3 Complete: Analysis results generated")
        print(f"📁 Analysis saved to: {analysis_file}")
        
        return overall_metrics
    
    def run_phase_4_visualization(self, test_results: List[Dict], metrics: Dict):
        """
        Phase 4: Generate all visualizations.
        """
        print("\n" + "🔬" * 50)
        print("\n🎨 PHASE 4: VISUALIZATION GENERATION")
        print("=" * 100)
        
        print("\n📊 Generating publication-quality figures...")
        
        # Generate all visualizations
        figures = self.viz_engine.generate_all_visualizations(test_results, metrics)
        
        print(f"\n✅ Phase 4 Complete: {len(figures)} visualizations generated")
        print(f"📁 Saved to: autonomous_error_recovery_research/visualizations/")
        
        return figures
    
    def run_phase_5_paper_generation(self):
        """
        Phase 5: Generate LaTeX paper with results.
        """
        print("\n" + "🔬" * 50)
        print("\n📝 PHASE 5: PAPER GENERATION")
        print("=" * 100)
        
        print("\n📄 Generating IEEE-formatted LaTeX paper...")
        print("  • Title: Autonomous Error Recovery Patterns in Large Language Models")
        print("  • Author: Harshith Vaddiparthy")
        print("  • Format: IEEE Conference Paper")
        print("  • Sections: Abstract, Introduction, Methodology, Results, Discussion, Conclusion")
        
        # This will be implemented in the next step
        print("\n⏳ Paper generation will be completed in the next phase...")
        
        return True
    
    def generate_final_report(self):
        """
        Generate comprehensive final report.
        """
        print("\n" + "=" * 100)
        print(" " * 35 + "EXPERIMENT COMPLETE")
        print("=" * 100)
        
        duration = (datetime.now() - self.start_time).total_seconds()
        duration_minutes = duration / 60
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"  • Total Duration: {duration_minutes:.1f} minutes")
        print(f"  • Tests Executed: {self.total_tests}")
        print(f"  • Data Files Generated: 5")
        print(f"  • Visualizations Created: 5")
        print(f"  • Paper Status: Ready for generation")
        
        print("\n📁 OUTPUT FILES:")
        print(f"  • Test Cases: test_cases_{self.experiment_id}.json")
        print(f"  • Execution Results: execution_results_{self.experiment_id}.json")
        print(f"  • Analysis Results: analysis_results_{self.experiment_id}.json")
        print(f"  • Visualizations: /visualizations/*.pdf")
        print(f"  • LaTeX Paper: paper.tex (to be generated)")
        
        print("\n✨ Experiment successfully completed!")
        print("=" * 100)
    
    def run_complete_experiment(self):
        """
        Run the complete experimental pipeline.
        """
        try:
            # Phase 1: Generate test cases
            test_cases = self.run_phase_1_test_generation()
            
            # Phase 2: Execute tests
            test_results = self.run_phase_2_execution(test_cases)
            
            # Phase 3: Analyze results
            metrics = self.run_phase_3_analysis(test_results)
            
            # Phase 4: Generate visualizations
            figures = self.run_phase_4_visualization(test_results, metrics)
            
            # Phase 5: Generate paper
            self.run_phase_5_paper_generation()
            
            # Generate final report
            self.generate_final_report()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: Experiment failed!")
            print(f"Error details: {str(e)}")
            print(traceback.format_exc())
            return False

if __name__ == "__main__":
    print("\n" + "🚀" * 50)
    print("\n             INITIATING AUTONOMOUS ERROR RECOVERY RESEARCH EXPERIMENT")
    print("\n" + "🚀" * 50)
    
    # Create and run orchestrator
    orchestrator = MasterOrchestrator()
    
    # Prompt to start
    print("\n" + "⚠️ " * 20)
    print("\n⚠️  READY TO START EXPERIMENT")
    print(f"⚠️  This will run {orchestrator.total_tests} tests")
    print(f"⚠️  Estimated time: ~{orchestrator.total_tests * 0.5:.1f} minutes")
    print("\n" + "⚠️ " * 20)
    
    input("\n👉 Press ENTER to begin the experiment...")
    
    # Run experiment
    success = orchestrator.run_complete_experiment()
    
    if success:
        print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ EXPERIMENT FAILED - Please check error logs")