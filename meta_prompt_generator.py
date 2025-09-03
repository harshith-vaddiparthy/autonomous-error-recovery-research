"""
Meta-Prompt Generator for Autonomous Error Recovery Research
Author: Harshith Vaddiparthy
Date: January 2025
Purpose: Generate systematic prompts to test LLM error recovery mechanisms
"""

import json
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib

class MetaPromptGenerator:
    """
    This class generates meta-prompts designed to systematically test
    error recovery patterns in Large Language Models.
    """
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_categories = self._initialize_error_categories()
        self.recovery_metrics = {
            "total_tests": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "partial_recoveries": 0,
            "recovery_times": [],
            "error_type_distribution": {}
        }
        
    def _initialize_error_categories(self) -> Dict[str, List[str]]:
        """
        Initialize comprehensive error categories for testing.
        Each category contains specific error patterns to inject.
        """
        return {
            "SYNTAX_ERRORS": [
                "missing_parenthesis",
                "unclosed_quotes",
                "invalid_indentation",
                "missing_colon",
                "bracket_mismatch",
                "missing_comma",
                "invalid_operator",
                "keyword_typo",
                "missing_import",
                "circular_import"
            ],
            "LOGIC_ERRORS": [
                "infinite_loop",
                "off_by_one",
                "incorrect_condition",
                "wrong_operator",
                "invalid_comparison",
                "missing_base_case",
                "incorrect_recursion",
                "race_condition",
                "deadlock_potential",
                "incorrect_algorithm"
            ],
            "TYPE_ERRORS": [
                "string_int_concatenation",
                "none_type_access",
                "list_dict_confusion",
                "invalid_cast",
                "missing_type_check",
                "incompatible_types",
                "null_pointer",
                "undefined_variable",
                "scope_violation",
                "immutable_mutation"
            ],
            "RUNTIME_ERRORS": [
                "division_by_zero",
                "index_out_of_bounds",
                "key_error",
                "file_not_found",
                "network_timeout",
                "memory_overflow",
                "stack_overflow",
                "resource_exhaustion",
                "permission_denied",
                "encoding_error"
            ],
            "SEMANTIC_ERRORS": [
                "incorrect_api_usage",
                "wrong_parameter_order",
                "missing_validation",
                "incorrect_return_type",
                "side_effect_violation",
                "contract_violation",
                "invariant_broken",
                "precondition_failed",
                "postcondition_failed",
                "incorrect_assumption"
            ]
        }
    
    def generate_error_injection_prompt(self, error_type: str, complexity: int = 1) -> str:
        """
        Generate a prompt that intentionally contains a specific error type.
        
        Args:
            error_type: The type of error to inject
            complexity: Complexity level (1-5) affecting error subtlety
            
        Returns:
            A prompt string with embedded error
        """
        base_prompt = self._get_base_prompt(error_type)
        error_injection = self._inject_error(base_prompt, error_type, complexity)
        
        meta_prompt = f"""
        [META-EXPERIMENTAL PROMPT - ERROR RECOVERY TEST]
        
        Task ID: {self._generate_task_id(error_type)}
        Error Category: {self._get_category(error_type)}
        Error Type: {error_type}
        Complexity Level: {complexity}
        Timestamp: {datetime.now().isoformat()}
        
        INSTRUCTION: The following code contains an intentional error. 
        Your task is to:
        1. Identify the error
        2. Explain why it's problematic
        3. Provide the corrected version
        4. Verify your solution works
        
        CODE WITH ERROR:
        ```python
        {error_injection}
        ```
        
        Please proceed with error detection and recovery.
        """
        
        return meta_prompt
    
    def _get_base_prompt(self, error_type: str) -> str:
        """
        Get base code template for specific error type.
        """
        base_templates = {
            "missing_parenthesis": """
def calculate_average(numbers):
    if len(numbers == 0:  # Error here
        return 0
    total = sum(numbers)
    return total / len(numbers)
            """,
            "infinite_loop": """
def process_items(items):
    i = 0
    while i < len(items):
        print(f"Processing {items[i]}")
        # Missing increment - infinite loop
    return "Done"
            """,
            "string_int_concatenation": """
def format_message(user_id, message):
    header = "User " + user_id + ": "  # Error: user_id is int
    return header + message

result = format_message(12345, "Hello World")
            """,
            "division_by_zero": """
def calculate_rate(successes, total):
    failure_count = total - successes
    success_rate = successes / total
    failure_rate = failure_count / (total - total)  # Division by zero
    return success_rate, failure_rate
            """,
            "incorrect_api_usage": """
import requests

def fetch_data(endpoint):
    response = requests.get(endpoint, verify=False, timeout=None)
    # Multiple issues: SSL verification disabled, no timeout
    data = response.text()  # Wrong: should be response.text (property not method)
    return json.loads(data)  # Missing import for json
            """
        }
        
        return base_templates.get(error_type, self._generate_generic_template(error_type))
    
    def _inject_error(self, code: str, error_type: str, complexity: int) -> str:
        """
        Inject specific error patterns into code based on complexity.
        """
        if complexity == 1:
            # Simple, obvious error
            return code
        elif complexity == 2:
            # Slightly hidden error
            return self._add_distraction_code(code)
        elif complexity == 3:
            # Error in nested structure
            return self._nest_error(code)
        elif complexity == 4:
            # Multiple related errors
            return self._add_cascading_errors(code)
        else:
            # Complex, subtle error
            return self._create_subtle_error(code)
    
    def _generate_task_id(self, error_type: str) -> str:
        """Generate unique task ID for tracking."""
        timestamp = str(time.time())
        hash_input = f"{error_type}_{timestamp}_{self.experiment_id}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _get_category(self, error_type: str) -> str:
        """Get the category for a given error type."""
        for category, errors in self.error_categories.items():
            if error_type in errors:
                return category
        return "UNKNOWN"
    
    def _generate_generic_template(self, error_type: str) -> str:
        """Generate a generic template when specific template not available."""
        return f"""
# Generic template for {error_type}
def example_function(param1, param2):
    # This is where the {error_type} will be injected
    result = param1 + param2
    return result
        """
    
    def _add_distraction_code(self, code: str) -> str:
        """Add distraction code to make error less obvious."""
        distraction = """
    # Additional processing
    temp_values = [1, 2, 3, 4, 5]
    processed = [x * 2 for x in temp_values]
        """
        lines = code.split('\n')
        insert_pos = len(lines) // 2
        lines.insert(insert_pos, distraction)
        return '\n'.join(lines)
    
    def _nest_error(self, code: str) -> str:
        """Nest the error within additional structure."""
        return f"""
def wrapper_function():
    try:
{code}
    except Exception as e:
        print(f"Error occurred: {{e}}")
        raise
        """
    
    def _add_cascading_errors(self, code: str) -> str:
        """Add multiple related errors that cascade."""
        return code + """
    # Additional code with cascading errors
    result = undefined_variable  # First error
    processed = result.split()  # Cascading error
    return processed[100]  # Another cascading error
        """
    
    def _create_subtle_error(self, code: str) -> str:
        """Create a subtle, hard-to-detect error."""
        # Replace == with = in conditions (assignment instead of comparison)
        import re
        subtle_code = re.sub(r'if\s+(.+?)\s*==\s*(.+?):', r'if \1 = \2:', code)
        return subtle_code

# Initialize the generator
generator = MetaPromptGenerator()
print("Meta-Prompt Generator initialized successfully!")
print(f"Experiment ID: {generator.experiment_id}")
print(f"Total error categories: {len(generator.error_categories)}")
print(f"Total error types: {sum(len(errors) for errors in generator.error_categories.values())}")