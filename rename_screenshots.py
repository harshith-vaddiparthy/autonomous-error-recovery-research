#!/usr/bin/env python3
"""
Screenshot Renaming Script for Autonomous Error Recovery Research
This script renames screenshots with proper descriptive names for the paper
"""

import os
import shutil
from datetime import datetime

def rename_screenshots():
    """
    Rename screenshots with descriptive names for the research paper.
    """
    
    # Define the screenshot naming scheme
    screenshot_names = [
        # Initial Setup & Framework
        "00_research_framework_initialization.png",
        "01_experimental_configuration_display.png",
        "02_meta_prompt_generator_code.png",
        "03_experiment_runner_architecture.png",
        "04_visualization_engine_setup.png",
        
        # Experimental Execution
        "05_experiment_start_header.png",
        "06_test_execution_phase1_error_injection.png",
        "07_test_execution_phase2_error_detection.png",
        "08_test_execution_phase3_error_correction.png",
        "09_test_execution_phase4_verification.png",
        
        # Progress Reports
        "10_progress_report_10_tests.png",
        "11_progress_report_30_tests.png",
        "12_progress_report_50_tests.png",
        "13_progress_report_100_tests.png",
        "14_progress_report_150_tests_complete.png",
        
        # Results & Analysis
        "15_overall_performance_metrics.png",
        "16_performance_by_error_type.png",
        "17_performance_by_complexity_level.png",
        "18_statistical_analysis_summary.png",
        "19_experiment_completion_report.png",
        
        # Data & Visualizations
        "20_json_data_structure.png",
        "21_error_recovery_heatmap.png",
        "22_recovery_time_distribution.png",
        "23_correlation_matrix.png",
        "24_complexity_analysis_graphs.png",
        "25_performance_dashboard.png",
        
        # Code Execution Evidence
        "26_python_script_execution.png",
        "27_test_case_generation.png",
        "28_automated_testing_pipeline.png",
        "29_data_collection_process.png",
        "30_final_results_summary.png"
    ]
    
    screenshots_dir = "autonomous_error_recovery_research/screenshots"
    
    # Create screenshots directory if it doesn't exist
    os.makedirs(screenshots_dir, exist_ok=True)
    
    print("=" * 80)
    print("SCREENSHOT RENAMING TOOL")
    print("=" * 80)
    print(f"\nScreenshots directory: {screenshots_dir}")
    print(f"Expected number of screenshots: {len(screenshot_names)}")
    print("\n" + "-" * 80)
    
    # Get all image files in the screenshots directory
    existing_files = [f for f in os.listdir(screenshots_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not existing_files:
        print("\n⚠️  No screenshots found in the directory!")
        print("\nPlease add your screenshots to:")
        print(f"  {os.path.abspath(screenshots_dir)}")
        print("\nThen run this script again.")
        return
    
    print(f"\nFound {len(existing_files)} screenshots to rename")
    
    # Sort existing files to maintain order
    existing_files.sort()
    
    # Create backup directory
    backup_dir = f"{screenshots_dir}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"\nCreating backup in: {backup_dir}")
    
    # Rename files
    renamed_count = 0
    for i, old_name in enumerate(existing_files):
        if i < len(screenshot_names):
            new_name = screenshot_names[i]
            old_path = os.path.join(screenshots_dir, old_name)
            new_path = os.path.join(screenshots_dir, new_name)
            backup_path = os.path.join(backup_dir, old_name)
            
            # Backup original
            shutil.copy2(old_path, backup_path)
            
            # Rename file
            if old_path != new_path:
                shutil.move(old_path, new_path)
                print(f"  ✓ Renamed: {old_name} → {new_name}")
                renamed_count += 1
            else:
                print(f"  - Skipped: {old_name} (already has correct name)")
        else:
            print(f"  ⚠️  Extra file: {old_name} (no mapping available)")
    
    print("\n" + "-" * 80)
    print(f"\n✅ Renaming complete!")
    print(f"  • Files renamed: {renamed_count}")
    print(f"  • Backup created: {backup_dir}")
    
    # Create index file
    index_file = os.path.join(screenshots_dir, "SCREENSHOT_INDEX.md")
    with open(index_file, 'w') as f:
        f.write("# Screenshot Index for Autonomous Error Recovery Research\n\n")
        f.write("## Screenshot Descriptions\n\n")
        
        categories = {
            "Framework Setup": (0, 4),
            "Experimental Execution": (5, 9),
            "Progress Reports": (10, 14),
            "Results & Analysis": (15, 19),
            "Data & Visualizations": (20, 25),
            "Code Execution Evidence": (26, 30)
        }
        
        for category, (start, end) in categories.items():
            f.write(f"\n### {category}\n\n")
            for i in range(start, min(end + 1, len(screenshot_names))):
                if i < len(existing_files):
                    f.write(f"- `{screenshot_names[i]}` - {screenshot_names[i].replace('.png', '').replace('_', ' ').title()}\n")
    
    print(f"  • Index created: {index_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    rename_screenshots()