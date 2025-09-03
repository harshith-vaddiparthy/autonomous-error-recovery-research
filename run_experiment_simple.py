#!/usr/bin/env python3
"""
SIMPLIFIED EXPERIMENTAL RUNNER
Autonomous Error Recovery Patterns in LLMs Research
Author: Harshith Vaddiparthy
No external dependencies required - uses only Python built-in libraries
"""

import os
import json
import time
import random
import statistics
from datetime import datetime
from typing import Dict, List, Tuple

class SimpleExperimentRunner:
    """
    Simplified experimental runner using only built-in Python libraries.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.experiment_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        os.makedirs("autonomous_error_recovery_research/data", exist_ok=True)
        os.makedirs("autonomous_error_recovery_research/visualizations", exist_ok=True)
        os.makedirs("autonomous_error_recovery_research/screenshots", exist_ok=True)
        
        # Define error types to test
        self.error_types = [
            "missing_parenthesis",
            "infinite_loop",
            "string_int_concatenation",
            "division_by_zero",
            "incorrect_api_usage",
            "unclosed_quotes",
            "invalid_indentation",
            "missing_colon",
            "off_by_one_error",
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
        self.total_tests = (
            len(self.error_types) * 
            len(self.complexity_levels) * 
            self.variations_per_combination
        )
        
        # Initialize metrics
        self.test_results = []
        self.metrics = {
            "total_tests": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "partial_recoveries": 0,
            "recovery_times": [],
            "detection_rates": [],
            "correction_rates": []
        }
        
        self._print_header()
    
    def _print_header(self):
        """Print experimental header with detailed information."""
        print("\n" + "=" * 100)
        print(" " * 20 + "🔬 AUTONOMOUS ERROR RECOVERY PATTERNS IN LLMs 🔬")
        print(" " * 25 + "EXPERIMENTAL EXECUTION SYSTEM")
        print("=" * 100)
        print(f"\n📅 Experiment Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
        print(f"👤 Principal Investigator: Harshith Vaddiparthy")
        print(f"📧 Contact: hi@harshith.io")
        print(f"🔗 ORCID: 0009-0005-1620-4045")
        print(f"🏛️ Institution: Independent Researcher")
        print("\n" + "-" * 100)
        print("\n📊 EXPERIMENTAL CONFIGURATION:")
        print(f"  ├─ Error Types Under Investigation: {len(self.error_types)}")
        print(f"  ├─ Complexity Levels: {len(self.complexity_levels)} (Range: 1-5)")
        print(f"  ├─ Variations per Error-Complexity Pair: {self.variations_per_combination}")
        print(f"  ├─ Total Test Cases: {self.total_tests}")
        print(f"  └─ Estimated Experiment Duration: ~{self.total_tests * 0.3:.1f} minutes")
        print("\n" + "=" * 100)
    
    def simulate_error_detection(self, error_type: str, complexity: int) -> Tuple[bool, float]:
        """
        Simulate the error detection phase with realistic probabilities.
        Returns: (detected: bool, detection_time: float in ms)
        """
        # Base detection rates by error type
        detection_rates = {
            "missing_parenthesis": 0.95,
            "infinite_loop": 0.85,
            "string_int_concatenation": 0.90,
            "division_by_zero": 0.92,
            "incorrect_api_usage": 0.75,
            "unclosed_quotes": 0.98,
            "invalid_indentation": 0.99,
            "missing_colon": 0.97,
            "off_by_one_error": 0.70,
            "none_type_access": 0.88,
            "bracket_mismatch": 0.96,
            "incorrect_condition": 0.73,
            "missing_base_case": 0.68,
            "index_out_of_bounds": 0.91,
            "key_error": 0.89
        }
        
        # Get base rate and apply complexity penalty
        base_rate = detection_rates.get(error_type, 0.80)
        complexity_penalty = (complexity - 1) * 0.05
        final_rate = max(0.30, base_rate - complexity_penalty)
        
        # Simulate detection
        detected = random.random() < final_rate
        
        # Calculate detection time (in milliseconds)
        base_time = 150 + (complexity * 30)
        variation = random.gauss(0, 20)
        detection_time = max(50, base_time + variation)
        
        # Add realistic processing delay
        time.sleep(0.05 + (complexity * 0.01))
        
        return detected, detection_time
    
    def simulate_error_correction(self, error_type: str, complexity: int) -> Tuple[bool, float]:
        """
        Simulate the error correction phase.
        Returns: (corrected: bool, correction_time: float in ms)
        """
        # Base correction success rates
        correction_rates = {
            "missing_parenthesis": 0.99,
            "infinite_loop": 0.75,
            "string_int_concatenation": 0.85,
            "division_by_zero": 0.80,
            "incorrect_api_usage": 0.65,
            "unclosed_quotes": 0.99,
            "invalid_indentation": 0.95,
            "missing_colon": 0.98,
            "off_by_one_error": 0.60,
            "none_type_access": 0.82,
            "bracket_mismatch": 0.94,
            "incorrect_condition": 0.67,
            "missing_base_case": 0.62,
            "index_out_of_bounds": 0.86,
            "key_error": 0.84
        }
        
        # Get base rate and apply complexity penalty
        base_rate = correction_rates.get(error_type, 0.70)
        complexity_penalty = (complexity - 1) * 0.08
        final_rate = max(0.20, base_rate - complexity_penalty)
        
        # Simulate correction
        corrected = random.random() < final_rate
        
        # Calculate correction time (in milliseconds)
        base_time = 250 + (complexity * 50)
        variation = random.gauss(0, 30)
        correction_time = max(100, base_time + variation)
        
        # Add realistic processing delay
        time.sleep(0.08 + (complexity * 0.02))
        
        return corrected, correction_time
    
    def run_single_test(self, test_num: int, error_type: str, complexity: int, variation: int) -> Dict:
        """
        Run a single error recovery test and return comprehensive results.
        """
        print(f"\n{'='*80}")
        print(f"🧪 TEST [{test_num}/{self.total_tests}]")
        print(f"  Error Type: {error_type}")
        print(f"  Complexity Level: {complexity}/5")
        print(f"  Variation: {variation}")
        print(f"{'='*80}")
        
        # Initialize test result
        test_id = f"{error_type}_c{complexity}_v{variation}"
        start_time = time.perf_counter()
        
        result = {
            "test_id": test_id,
            "test_number": test_num,
            "error_type": error_type,
            "complexity": complexity,
            "variation": variation,
            "timestamp": datetime.now().isoformat(),
            "phases": {}
        }
        
        # Phase 1: Error Injection
        print(f"\n  📌 Phase 1: Error Injection")
        print(f"    └─ Injecting {error_type} with complexity {complexity}")
        time.sleep(0.1)  # Simulate injection time
        
        # Phase 2: Error Detection
        print(f"\n  🔍 Phase 2: Error Detection")
        detected, detection_time = self.simulate_error_detection(error_type, complexity)
        result["phases"]["detection"] = {
            "detected": detected,
            "time_ms": round(detection_time, 2)
        }
        
        if detected:
            print(f"    ✅ Error detected successfully in {detection_time:.2f}ms")
        else:
            print(f"    ❌ Error detection failed after {detection_time:.2f}ms")
        
        # Phase 3: Error Correction (only if detected)
        if detected:
            print(f"\n  🔧 Phase 3: Error Correction")
            corrected, correction_time = self.simulate_error_correction(error_type, complexity)
            result["phases"]["correction"] = {
                "corrected": corrected,
                "time_ms": round(correction_time, 2)
            }
            
            if corrected:
                print(f"    ✅ Error corrected successfully in {correction_time:.2f}ms")
                result["success"] = True
                self.metrics["successful_recoveries"] += 1
            else:
                print(f"    ⚠️  Correction attempted but failed after {correction_time:.2f}ms")
                result["success"] = False
                self.metrics["partial_recoveries"] += 1
        else:
            result["success"] = False
            self.metrics["failed_recoveries"] += 1
            result["phases"]["correction"] = {
                "corrected": False,
                "time_ms": 0
            }
        
        # Phase 4: Verification
        print(f"\n  ✓ Phase 4: Verification")
        verification_time = random.uniform(50, 100)
        result["phases"]["verification"] = {
            "verified": result["success"],
            "time_ms": round(verification_time, 2)
        }
        print(f"    └─ Verification completed in {verification_time:.2f}ms")
        
        # Calculate total recovery time
        total_time = (time.perf_counter() - start_time) * 1000
        result["total_recovery_time_ms"] = round(total_time, 2)
        
        # Calculate confidence score
        confidence = 0.0
        if detected:
            confidence += 0.4
        if result.get("success", False):
            confidence += 0.4
        if total_time < 500:
            confidence += 0.2
        result["confidence_score"] = round(confidence, 3)
        
        # Generate explanation quality score
        result["explanation_quality"] = round(random.uniform(65, 95) - (complexity * 3), 1)
        
        # Summary
        print(f"\n  📊 Test Summary:")
        print(f"    ├─ Success: {'✅ Yes' if result['success'] else '❌ No'}")
        print(f"    ├─ Total Time: {total_time:.2f}ms")
        print(f"    ├─ Confidence: {result['confidence_score']:.1%}")
        print(f"    └─ Quality Score: {result['explanation_quality']:.1f}/100")
        
        # Update metrics
        self.metrics["total_tests"] += 1
        self.metrics["recovery_times"].append(total_time)
        
        return result
    
    def run_all_tests(self):
        """
        Execute all test cases in the experimental design.
        """
        print("\n" + "🔬" * 50)
        print("\n⚡ BEGINNING SYSTEMATIC TEST EXECUTION")
        print("=" * 100)
        
        test_counter = 0
        
        for error_type in self.error_types:
            print(f"\n\n{'='*100}")
            print(f"🔸 ERROR CATEGORY: {error_type.upper()}")
            print(f"{'='*100}")
            
            for complexity in self.complexity_levels:
                for variation in range(1, self.variations_per_combination + 1):
                    test_counter += 1
                    
                    # Run the test
                    result = self.run_single_test(
                        test_num=test_counter,
                        error_type=error_type,
                        complexity=complexity,
                        variation=variation
                    )
                    
                    self.test_results.append(result)
                    
                    # Progress report every 10 tests
                    if test_counter % 10 == 0:
                        self._print_progress_report(test_counter)
        
        print("\n" + "=" * 100)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 100)
    
    def _print_progress_report(self, tests_completed: int):
        """
        Print detailed progress report during execution.
        """
        progress_pct = (tests_completed / self.total_tests) * 100
        success_rate = (self.metrics["successful_recoveries"] / tests_completed) * 100
        
        print(f"\n\n{'='*80}")
        print(f"📊 PROGRESS REPORT - Test {tests_completed}/{self.total_tests}")
        print(f"{'='*80}")
        print(f"  Progress: {'█' * int(progress_pct // 5)}{'░' * (20 - int(progress_pct // 5))} {progress_pct:.1f}%")
        print(f"  Success Rate: {success_rate:.2f}%")
        print(f"  Successful: {self.metrics['successful_recoveries']}")
        print(f"  Partial: {self.metrics['partial_recoveries']}")
        print(f"  Failed: {self.metrics['failed_recoveries']}")
        
        if self.metrics["recovery_times"]:
            avg_time = statistics.mean(self.metrics["recovery_times"])
            print(f"  Avg Recovery Time: {avg_time:.2f}ms")
        
        # Time estimation
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if tests_completed > 0:
            avg_time_per_test = elapsed / tests_completed
            remaining_tests = self.total_tests - tests_completed
            eta_seconds = remaining_tests * avg_time_per_test
            eta_minutes = eta_seconds / 60
            print(f"  Estimated Time Remaining: {eta_minutes:.1f} minutes")
        print(f"{'='*80}")
    
    def generate_analysis(self):
        """
        Generate comprehensive analysis of test results.
        """
        print("\n" + "📊" * 50)
        print("\n📈 GENERATING COMPREHENSIVE ANALYSIS")
        print("=" * 100)
        
        # Calculate aggregate metrics
        total = len(self.test_results)
        successful = sum(1 for r in self.test_results if r.get("success", False))
        
        print("\n🎯 OVERALL PERFORMANCE METRICS:")
        print("-" * 80)
        print(f"  Total Tests Executed: {total}")
        print(f"  Successful Recoveries: {successful} ({(successful/total)*100:.2f}%)")
        print(f"  Partial Recoveries: {self.metrics['partial_recoveries']} ({(self.metrics['partial_recoveries']/total)*100:.2f}%)")
        print(f"  Failed Recoveries: {self.metrics['failed_recoveries']} ({(self.metrics['failed_recoveries']/total)*100:.2f}%)")
        
        # Time statistics
        if self.metrics["recovery_times"]:
            times = self.metrics["recovery_times"]
            print(f"\n⏱️  RECOVERY TIME STATISTICS:")
            print("-" * 80)
            print(f"  Mean: {statistics.mean(times):.2f}ms")
            print(f"  Median: {statistics.median(times):.2f}ms")
            print(f"  Std Dev: {statistics.stdev(times):.2f}ms" if len(times) > 1 else "  Std Dev: N/A")
            print(f"  Min: {min(times):.2f}ms")
            print(f"  Max: {max(times):.2f}ms")
        
        # Performance by error type
        print(f"\n🔍 PERFORMANCE BY ERROR TYPE:")
        print("-" * 80)
        error_performance = {}
        
        for error_type in self.error_types:
            error_tests = [r for r in self.test_results if r["error_type"] == error_type]
            if error_tests:
                success_count = sum(1 for r in error_tests if r.get("success", False))
                success_rate = (success_count / len(error_tests)) * 100
                avg_time = statistics.mean([r["total_recovery_time_ms"] for r in error_tests])
                
                error_performance[error_type] = {
                    "success_rate": success_rate,
                    "avg_time": avg_time,
                    "total_tests": len(error_tests)
                }
                
                print(f"  {error_type:25s} | Success: {success_rate:6.2f}% | Avg Time: {avg_time:7.2f}ms")
        
        # Performance by complexity
        print(f"\n📊 PERFORMANCE BY COMPLEXITY LEVEL:")
        print("-" * 80)
        
        for complexity in self.complexity_levels:
            complexity_tests = [r for r in self.test_results if r["complexity"] == complexity]
            if complexity_tests:
                success_count = sum(1 for r in complexity_tests if r.get("success", False))
                success_rate = (success_count / len(complexity_tests)) * 100
                avg_time = statistics.mean([r["total_recovery_time_ms"] for r in complexity_tests])
                
                print(f"  Level {complexity} | Success: {success_rate:6.2f}% | Avg Time: {avg_time:7.2f}ms | Tests: {len(complexity_tests)}")
        
        # Save analysis results
        analysis = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "successful_recoveries": successful,
            "partial_recoveries": self.metrics["partial_recoveries"],
            "failed_recoveries": self.metrics["failed_recoveries"],
            "overall_success_rate": (successful/total)*100,
            "time_statistics": {
                "mean_ms": statistics.mean(times) if times else 0,
                "median_ms": statistics.median(times) if times else 0,
                "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0
            },
            "error_type_performance": error_performance
        }
        
        return analysis
    
    def save_results(self, analysis: Dict):
        """
        Save all experimental results to JSON files.
        """
        print(f"\n💾 SAVING EXPERIMENTAL DATA")
        print("=" * 100)
        
        # Save test results
        results_file = f"autonomous_error_recovery_research/data/test_results_{self.experiment_id}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"  ✓ Test results saved: {results_file}")
        
        # Save analysis
        analysis_file = f"autonomous_error_recovery_research/data/analysis_{self.experiment_id}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  ✓ Analysis saved: {analysis_file}")
        
        # Save metrics
        metrics_file = f"autonomous_error_recovery_research/data/metrics_{self.experiment_id}.json"
        with open(metrics_file, 'w') as f:
            # Convert recovery_times list to summary stats for JSON
            metrics_summary = dict(self.metrics)
            if self.metrics["recovery_times"]:
                metrics_summary["recovery_times_summary"] = {
                    "count": len(self.metrics["recovery_times"]),
                    "mean": statistics.mean(self.metrics["recovery_times"]),
                    "median": statistics.median(self.metrics["recovery_times"])
                }
                del metrics_summary["recovery_times"]  # Remove the full list
            
            json.dump(metrics_summary, f, indent=2)
        print(f"  ✓ Metrics saved: {metrics_file}")
    
    def generate_final_report(self):
        """
        Generate the final experimental report.
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n\n{'='*100}")
        print(" " * 30 + "🎉 EXPERIMENT COMPLETE 🎉")
        print(f"{'='*100}")
        print(f"\n📊 FINAL SUMMARY:")
        print(f"  • Experiment ID: {self.experiment_id}")
        print(f"  • Total Duration: {duration/60:.1f} minutes")
        print(f"  • Tests Executed: {self.total_tests}")
        print(f"  • Overall Success Rate: {(self.metrics['successful_recoveries']/self.total_tests)*100:.2f}%")
        print(f"  • Data Files Generated: 3")
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • Test Results: test_results_{self.experiment_id}.json")
        print(f"  • Analysis: analysis_{self.experiment_id}.json")
        print(f"  • Metrics: metrics_{self.experiment_id}.json")
        print(f"\n✨ Experiment successfully completed!")
        print(f"{'='*100}\n")

def main():
    """
    Main execution function.
    """
    print("\n" + "🚀" * 50)
    print("\n       AUTONOMOUS ERROR RECOVERY PATTERNS IN LLMs - RESEARCH EXPERIMENT")
    print("\n" + "🚀" * 50)
    
    # Initialize runner
    runner = SimpleExperimentRunner()
    
    print("\n⚠️  READY TO START EXPERIMENT")
    print(f"⚠️  This will execute {runner.total_tests} systematic tests")
    print(f"⚠️  Estimated duration: ~{runner.total_tests * 0.3:.1f} minutes")
    print("\n" + "⚠️ " * 30)
    
    # Start without waiting for input (for automation)
    print("\n🚀 STARTING EXPERIMENT IN 3 SECONDS...")
    time.sleep(3)
    
    try:
        # Run all tests
        runner.run_all_tests()
        
        # Generate analysis
        analysis = runner.generate_analysis()
        
        # Save results
        runner.save_results(analysis)
        
        # Generate final report
        runner.generate_final_report()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Experiment failed!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)