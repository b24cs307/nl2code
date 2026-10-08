"""
Utility functions for file management, filename generation, syntax validation,
and terminal formatting.
"""
import ast
import os
import re
import sys
import time
from typing import Tuple, Dict, Any, Optional


STOP_WORDS = {
    "write", "a", "an", "the", "function", "that", "to", "in", "and", "or",
    "for", "of", "is", "given", "returns", "return", "takes", "which", "with",
    "how", "code", "python", "script", "program", "creates", "create", "calculates",
    "checks", "check", "computes", "compute"
}


def prompt_to_filename(prompt: str, extension: str = ".py") -> str:
    """
    Generates a clean snake_case filename based on the prompt description.

    Args:
        prompt: Natural language task description.
        extension: File extension (default '.py').

    Returns:
        snake_case string suitable for python filename.
    """
    # Lowercase and keep alphanumeric + spaces
    words = re.sub(r'[^a-zA-Z0-9\s]', '', prompt.lower()).split()
    
    # Filter out common stop words if possible to make filename meaningful
    filtered = [w for w in words if w not in STOP_WORDS]
    if not filtered:
        filtered = words[:5]
    else:
        filtered = filtered[:6]

    slug = "_".join(filtered)
    if not slug:
        slug = "generated_code"

    return f"{slug}{extension}"


def save_code_to_file(code: str, filepath: Optional[str] = None, prompt: str = "") -> str:
    """
    Saves python code string to a file. Auto-generates filename if not provided.

    Args:
        code: Python code content to write.
        filepath: Target file path or name.
        prompt: Task prompt used for auto-naming if filepath is empty.

    Returns:
        Absolute path to saved file.
    """
    if not filepath:
        out_dir = os.path.join(os.getcwd(), "generated")
        os.makedirs(out_dir, exist_ok=True)
        filename = prompt_to_filename(prompt)
        filepath = os.path.join(out_dir, filename)
    else:
        # Create directory if provided in path
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    abs_path = os.path.abspath(filepath)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")

    return abs_path


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """
    Validates if the provided code string is syntactically valid Python.

    Args:
        code: Python code string.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if not code or not code.strip():
        return False, "Code is empty"

    try:
        ast.parse(code)
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"Syntax Error at line {e.lineno}, col {e.offset}: {e.msg}"


def execute_generated_code(code: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Executes generated code in an isolated local dictionary scope to verify
    it compiles and defines expected functions without errors.

    Args:
        code: Python code string.

    Returns:
        Tuple of (success: bool, output_message: str, execution_scope: dict).
    """
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope)
        functions = [name for name, obj in scope.items() if callable(obj) and not name.startswith("__")]
        msg = f"Executed successfully. Defined functions: {', '.join(functions) if functions else 'None'}"
        return True, msg, scope
    except Exception as e:
        return False, f"Execution error: {type(e).__name__}: {str(e)}", scope


# Terminal Formatting & Colors
class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorize(text: str, color_code: str) -> str:
    """Wrap text in ANSI color codes if supported."""
    return f"{color_code}{text}{TerminalColors.ENDC}"
