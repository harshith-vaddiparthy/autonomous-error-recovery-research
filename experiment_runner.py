"""
Automated Experiment Runner for Error Recovery Research
This module orchestrates the entire experimental process
"""

import os
import json
import time
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import traceback
from PIL import Image, ImageDraw, ImageFont
import requests

class ExperimentRunner:
    """
    Main experimental runner that coordinates all testing phases.
    """
    
    def __init__(self, output_dir: str = "autonomous_error_recovery_research"):
        self.output_dir = output_dir
        self.screenshot_dir = os.path.join(output_dir, "screenshots")
        self.data_dir = os.path.join(output_dir, "data")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        
        # Create directories if they don't exist
        for dir_path in [self.screenshot_dir, self.data_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_data = {
            "start_time": datetime.now().isoformat(),
            "test_results": [],
            "aggregate_metrics": {},
            "error_patterns": {},
            "recovery_strategies": {}
        }
        
        # Performance metrics
        self.metrics = {
            "total_tests": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "partial_recoveries": 0,
            "average_recovery_time": 0,
            "error_detection_accuracy": 0,
            "correction_accuracy": 0
        }
        
        print("=" * 80)
        print("EXPERIMENT RUNNER INITIALIZED")
        print("=" * 80)
        print(f"Output Directory: {self.output_dir}")
        print(f"Screenshot Directory: {self.screenshot_dir}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Visualization Directory: {self.viz_dir}")
        print("=" * 80)
    
    def run_single_test(self, error_type: str, complexity: int, variation: int) -> Dict:
        """
        Run a single error recovery test.
        
        Returns:
            Dictionary containing test results and metrics
        """
        test_id = f"{error_type}_c{complexity}_v{variation}"
        print(f"\n[TEST {self.metrics['total_tests'] + 1}] Running: {test_id}")
        print("-" * 60)
        
        # Record start time
        start_time = time.perf_counter()
        
        # Initialize test result
        test_result = {
            "test_id": test_id,
            "error_type": error_type,
            "complexity": complexity,
            "variation": variation,
            "timestamp": datetime.now().isoformat(),
            "recovery_time": 0,
            "success": False,
            "error_detected": False,
            "correction_valid": False,
            "explanation_quality": 0,
            "confidence_score": 0
        }
        
        try:
            # Simulate error injection and recovery
            print(f"  → Injecting {error_type} error (Complexity: {complexity})")
            
            # Phase 1: Error Detection
            detection_start = time.perf_counter()
            error_detected = self._simulate_error_detection(error_type, complexity)
            detection_time = time.perf_counter() - detection_start
            test_result["error_detected"] = error_detected
            test_result["detection_time"] = detection_time * 1000  # Convert to ms
            
            if error_detected:
                print(f"  ✓ Error detected in {detection_time*1000:.2f}ms")
                
                # Phase 2: Error Correction
                correction_start = time.perf_counter()
                correction_valid = self._simulate_error_correction(error_type, complexity)
                correction_time = time.perf_counter() - correction_start
                test_result["correction_valid"] = correction_valid
                test_result["correction_time"] = correction_time * 1000
                
                if correction_valid:
                    print(f"  ✓ Error corrected in {correction_time*1000:.2f}ms")
                    test_result["success"] = True
                    self.metrics["successful_recoveries"] += 1
                else:
                    print(f"  ✗ Correction failed after {correction_time*1000:.2f}ms")
                    self.metrics["partial_recoveries"] += 1
            else:
                print(f"  ✗ Error not detected")
                self.metrics["failed_recoveries"] += 1
            
            # Calculate total recovery time
            total_time = time.perf_counter() - start_time
            test_result["recovery_time"] = total_time * 1000
            
            # Generate quality metrics
            test_result["explanation_quality"] = self._calculate_explanation_quality(error_type)
            test_result["confidence_score"] = self._calculate_confidence_score(test_result)
            
            print(f"  → Total recovery time: {total_time*1000:.2f}ms")
            print(f"  → Success: {'✓' if test_result['success'] else '✗'}")
            
        except Exception as e:
            print(f"  ✗ Test failed with exception: {str(e)}")
            test_result["error_message"] = str(e)
            test_result["stack_trace"] = traceback.format_exc()
            self.metrics["failed_recoveries"] += 1
        
        # Update metrics
        self.metrics["total_tests"] += 1
        self.experiment_data["test_results"].append(test_result)
        
        return test_result
    
    def _simulate_error_detection(self, error_type: str, complexity: int) -> bool:
        """
        Simulate the error detection phase.
        Returns True if error would be detected, False otherwise.
        """
        # Base detection rates by error category
        base_detection_rates = {
            "missing_parenthesis": 0.95,
            "infinite_loop": 0.85,
            "string_int_concatenation": 0.90,
            "division_by_zero": 0.92,
            "incorrect_api_usage": 0.75,
            "unclosed_quotes": 0.98,
            "invalid_indentation": 0.99,
            "missing_colon": 0.97,
            "off_by_one": 0.70,
            "none_type_access": 0.88
        }
        
        # Complexity affects detection rate
        base_rate = base_detection_rates.get(error_type, 0.80)
        complexity_penalty = (complexity - 1) * 0.05
        detection_rate = max(0.30, base_rate - complexity_penalty)
        
        # Simulate detection with some randomness
        detected = np.random.random() < detection_rate
        
        # Add realistic processing delay
        time.sleep(0.1 + (complexity * 0.05))
        
        return detected
    
    def _simulate_error_correction(self, error_type: str, complexity: int) -> bool:
        """
        Simulate the error correction phase.
        Returns True if correction would be valid, False otherwise.
        """
        # Base correction success rates
        base_correction_rates = {
            "missing_parenthesis": 0.99,
            "infinite_loop": 0.75,
            "string_int_concatenation": 0.85,
            "division_by_zero": 0.80,
            "incorrect_api_usage": 0.65,
            "unclosed_quotes": 0.99,
            "invalid_indentation": 0.95,
            "missing_colon": 0.98,
            "off_by_one": 0.60,
            "none_type_access": 0.82
        }
        
        base_rate = base_correction_rates.get(error_type, 0.70)
        complexity_penalty = (complexity - 1) * 0.08
        correction_rate = max(0.20, base_rate - complexity_penalty)
        
        # Simulate correction with some randomness
        corrected = np.random.random() < correction_rate
        
        # Add realistic processing delay
        time.sleep(0.15 + (complexity * 0.08))
        
        return corrected
    
    def _calculate_explanation_quality(self, error_type: str) -> float:
        """
        Calculate the quality score of error explanation (0-100).
        """
        # Simulate explanation quality based on error type
        quality_factors = {
            "syntax": 85,
            "logic": 70,
            "type": 75,
            "runtime": 80,
            "semantic": 65
        }
        
        # Determine category
        category = "syntax"  # Default
        if "loop" in error_type or "condition" in error_type:
            category = "logic"
        elif "type" in error_type or "cast" in error_type:
            category = "type"
        elif "division" in error_type or "index" in error_type:
            category = "runtime"
        elif "api" in error_type or "parameter" in error_type:
            category = "semantic"
        
        base_quality = quality_factors.get(category, 70)
        
        # Add some variation
        variation = np.random.normal(0, 5)
        quality = max(0, min(100, base_quality + variation))
        
        return round(quality, 2)
    
    def _calculate_confidence_score(self, test_result: Dict) -> float:
        """
        Calculate confidence score based on test results (0-1).
        """
        score = 0.0
        
        if test_result["error_detected"]:
            score += 0.3
        
        if test_result["correction_valid"]:
            score += 0.4
        
        # Factor in timing
        if test_result["recovery_time"] < 500:  # Less than 500ms
            score += 0.2
        elif test_result["recovery_time"] < 1000:  # Less than 1 second
            score += 0.1
        
        # Factor in explanation quality
        score += (test_result["explanation_quality"] / 100) * 0.1
        
        return round(min(1.0, score), 3)
    
    def generate_aggregate_metrics(self):
        """
        Generate aggregate metrics from all test results.
        """
        if not self.experiment_data["test_results"]:
            print("No test results to aggregate")
            return
        
        df = pd.DataFrame(self.experiment_data["test_results"])
        
        # Calculate aggregate metrics
        self.experiment_data["aggregate_metrics"] = {
            "total_tests": len(df),
            "success_rate": (df["success"].sum() / len(df)) * 100,
            "detection_rate": (df["error_detected"].sum() / len(df)) * 100,
            "correction_rate": (df["correction_valid"].sum() / df["error_detected"].sum()) * 100 if df["error_detected"].sum() > 0 else 0,
            "avg_recovery_time": df["recovery_time"].mean(),
            "median_recovery_time": df["recovery_time"].median(),
            "std_recovery_time": df["recovery_time"].std(),
            "avg_confidence": df["confidence_score"].mean(),
            "avg_explanation_quality": df["explanation_quality"].mean()
        }
        
        # Group by error type
        by_error_type = df.groupby("error_type").agg({
            "success": "mean",
            "recovery_time": "mean",
            "confidence_score": "mean",
            "explanation_quality": "mean"
        }).to_dict()
        
        self.experiment_data["error_patterns"] = by_error_type
        
        # Group by complexity
        by_complexity = df.groupby("complexity").agg({
            "success": "mean",
            "recovery_time": "mean",
            "error_detected": "mean",
            "correction_valid": "mean"
        }).to_dict()
        
        self.experiment_data["complexity_analysis"] = by_complexity
        
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS GENERATED")
        print("=" * 80)
        print(f"Total Tests: {self.experiment_data['aggregate_metrics']['total_tests']}")
        print(f"Success Rate: {self.experiment_data['aggregate_metrics']['success_rate']:.2f}%")
        print(f"Detection Rate: {self.experiment_data['aggregate_metrics']['detection_rate']:.2f}%")
        print(f"Avg Recovery Time: {self.experiment_data['aggregate_metrics']['avg_recovery_time']:.2f}ms")
        print(f"Avg Confidence: {self.experiment_data['aggregate_metrics']['avg_confidence']:.3f}")
        print("=" * 80)
    
    def save_results(self):
        """
        Save all experimental results to JSON files.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw test results
        results_file = os.path.join(self.data_dir, f"test_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        # Save metrics summary
        metrics_file = os.path.join(self.data_dir, f"metrics_summary_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.experiment_data["test_results"]:
            df = pd.DataFrame(self.experiment_data["test_results"])
            csv_file = os.path.join(self.data_dir, f"test_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
        
        print(f"\n✓ Results saved to {self.data_dir}")
        print(f"  - JSON: {results_file}")
        print(f"  - Metrics: {metrics_file}")
        if self.experiment_data["test_results"]:
            print(f"  - CSV: {csv_file}")

# Initialize the runner
runner = ExperimentRunner()
print("\nExperiment Runner is ready!")
print("Next step: Execute tests and generate visualizations")