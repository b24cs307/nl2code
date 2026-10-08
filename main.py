"""
NL2Code CLI Entrypoint
Translates natural language prompts into executable Python code using Groq, Ollama, or Mock LLM providers.
"""

import argparse
import sys
import os

# Fix Unicode printing on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nl2code.providers import get_provider
from nl2code.fence_cleaner import clean_llm_code_response
from nl2code.prompts import SYSTEM_PROMPT
from nl2code.utils import (
    validate_python_syntax,
    execute_generated_code,
    save_code_to_file,
    colorize,
    TerminalColors
)


def run_nl2code(
    prompt: str,
    provider_type: str = "mock",
    api_key: str = None,
    model: str = None,
    output_path: str = None,
    execute: bool = True
):
    print(colorize(f"\n🚀 [NL2Code Generator] Provider: {provider_type.upper()}", TerminalColors.HEADER))
    print(colorize(f"📝 Prompt: '{prompt}'\n", TerminalColors.OKCYAN))

    try:
        provider = get_provider(provider_name=provider_type, api_key=api_key, model=model)
    except Exception as e:
        print(colorize(f"❌ Failed to initialize provider: {e}", TerminalColors.FAIL))
        sys.exit(1)

    print(colorize(f"📡 Generating code via {provider.provider_name}...", TerminalColors.OKBLUE))
    raw_response = provider.generate_code(prompt, SYSTEM_PROMPT)

    cleaned_code = clean_llm_code_response(raw_response)

    print(colorize("\n✨ --- Generated Python Code ---", TerminalColors.BOLD))
    print(cleaned_code)
    print(colorize("---------------------------------\n", TerminalColors.BOLD))

    # Syntax Validation
    is_valid, syntax_msg = validate_python_syntax(cleaned_code)
    if is_valid:
        print(colorize(f"✅ Syntax Check: {syntax_msg}", TerminalColors.OKGREEN))
    else:
        print(colorize(f"❌ Syntax Check Failed: {syntax_msg}", TerminalColors.FAIL))
        return

    # File saving
    saved_path = save_code_to_file(cleaned_code, filepath=output_path, prompt=prompt)
    print(colorize(f"💾 Code saved to: {saved_path}", TerminalColors.OKBLUE))

    # Optional Execution
    if execute:
        print(colorize("\n⚡ Executing generated code in sandbox...", TerminalColors.OKCYAN))
        success, exec_msg, scope = execute_generated_code(cleaned_code)
        if success:
            print(colorize(f"🎉 Execution Result: {exec_msg}", TerminalColors.OKGREEN))
        else:
            print(colorize(f"💥 Execution Failed: {exec_msg}", TerminalColors.FAIL))


def main():
    parser = argparse.ArgumentParser(description="Natural Language to Python Code Generator")
    parser.add_argument(
        "--prompt", "-p", type=str, default="Write a function to check if a number is prime",
        help="Natural language description of the Python task"
    )
    parser.add_argument(
        "--provider", "-pr", type=str, choices=["groq", "ollama", "mock"], default="mock",
        help="LLM provider backend (default: mock)"
    )
    parser.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="API Key (required for Groq unless GROQ_API_KEY environment variable is set)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="LLM Model name (e.g., llama-3.3-70b-versatile for Groq, llama3.2 for Ollama)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path to save code (auto-generated if omitted)"
    )
    parser.add_argument(
        "--no-exec", action="store_true",
        help="Skip automatic sandbox execution of generated code"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run interactive test suite demonstrating multiple prompts and providers"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_nl2code(
            prompt=args.prompt,
            provider_type=args.provider,
            api_key=args.api_key,
            model=args.model,
            output_path=args.output,
            execute=not args.no_exec
        )


def run_demo():
    print(colorize("================================================", TerminalColors.HEADER))
    print(colorize("   NL2Code Generator - Full System Test Run     ", TerminalColors.HEADER))
    print(colorize("================================================", TerminalColors.HEADER))

    sample_prompts = [
        "Write a function that checks if a number is prime",
        "Write a function to compute the factorial of a non-negative integer",
        "Write a function to count word frequencies in a text string",
        "Write a function to flatten a nested list of items"
    ]

    for idx, prompt in enumerate(sample_prompts, 1):
        print(colorize(f"\n--- Demo Test #{idx} ---", TerminalColors.WARNING))
        run_nl2code(prompt=prompt, provider_type="mock", execute=True)


if __name__ == "__main__":
    main()
