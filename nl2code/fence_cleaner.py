"""
Post-processing logic to clean LLM output responses.
Strips markdown code fences and unwanted conversational text.
"""
import re


def clean_llm_code_response(response_text: str) -> str:
    """
    Cleans LLM response text by removing markdown code fences and extraneous
    conversational text to ensure only valid, executable Python code remains.

    Args:
        response_text: Raw output string from LLM provider.

    Returns:
        Cleaned string containing strictly Python code.
    """
    if not response_text:
        return ""

    text = response_text.strip()

    # Pattern 1: Match code block enclosed in triple backticks with optional language specifier
    # e.g., ```python\n<code>\n```
    code_block_pattern = re.compile(
        r'```(?:python|py)?\s*\n(.*?)\n\s*```', re.DOTALL | re.IGNORECASE
    )

    matches = code_block_pattern.findall(text)
    if matches:
        # If one or more fenced code blocks are found, extract the code contents
        cleaned = "\n\n".join(match.strip() for match in matches if match.strip())
        return cleaned

    # Pattern 2: Single line fences or unclosed ``` at start/end
    # Remove leading ```python or ```py or ```
    text = re.sub(r'^```(?:python|py)?\s*\n?', '', text, flags=re.IGNORECASE)
    # Remove trailing ```
    text = re.sub(r'\n?\s*```$', '', text)

    # Pattern 3: Remove conversational prefixes before actual python code if any exists
    # Look for common intro phrases like "Here is the code:", "Here is the Python function:"
    lines = text.splitlines()
    code_start_idx = 0

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Common code starting indicators in Python
        if (
            stripped_line.startswith(("import ", "from ", "def ", "class ", "#", "if ", "\"\"\"", "'''"))
            or (":" in stripped_line and not stripped_line.endswith("."))
        ):
            code_start_idx = i
            break

    # Rejoin from code start line
    cleaned_lines = lines[code_start_idx:]
    
    # Remove trailing conversational lines after code end (e.g. "Hope this helps!", "Explanation:")
    code_end_idx = len(cleaned_lines)
    for i in range(len(cleaned_lines) - 1, -1, -1):
        line = cleaned_lines[i].strip()
        if line and not line.startswith(("Note:", "Hope", "Explanation:", "Usage:", "Example:")):
            code_end_idx = i + 1
            break

    final_code = "\n".join(cleaned_lines[:code_end_idx]).strip()

    return final_code
