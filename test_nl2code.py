"""
Unit Test Suite for NL2Code Generator
"""
import os
import unittest
from nl2code.providers import MockProvider, GroqProvider, OllamaProvider, get_provider
from nl2code.fence_cleaner import clean_llm_code_response
from nl2code.prompts import SYSTEM_PROMPT
from nl2code.utils import prompt_to_filename, validate_python_syntax, execute_generated_code


class TestFenceCleaner(unittest.TestCase):

    def test_markdown_fence_removal(self):
        raw = "```python\ndef add(a, b):\n    return a + b\n```"
        expected = "def add(a, b):\n    return a + b"
        self.assertEqual(clean_llm_code_response(raw), expected)

    def test_conversational_text_stripping(self):
        raw = "Here is your python function:\n```python\ndef greet(name):\n    return f'Hello {name}'\n```\nHope this helps!"
        cleaned = clean_llm_code_response(raw)
        self.assertTrue(cleaned.startswith("def greet(name):"))
        self.assertNotIn("Here is your python function", cleaned)
        self.assertNotIn("Hope this helps", cleaned)

    def test_raw_code_passthrough(self):
        raw = "def square(x):\n    return x ** 2"
        self.assertEqual(clean_llm_code_response(raw), raw)


class TestMockProvider(unittest.TestCase):

    def setUp(self):
        self.provider = MockProvider()

    def test_known_prompt_prime(self):
        code = self.provider.generate_code("Write a prime function", SYSTEM_PROMPT)
        cleaned = clean_llm_code_response(code)
        valid, msg = validate_python_syntax(cleaned)
        self.assertTrue(valid, f"Syntax error: {msg}")
        self.assertIn("def is_prime", cleaned)

    def test_known_prompt_factorial(self):
        code = self.provider.generate_code("Calculate factorial", SYSTEM_PROMPT)
        cleaned = clean_llm_code_response(code)
        valid, msg = validate_python_syntax(cleaned)
        self.assertTrue(valid, f"Syntax error: {msg}")
        self.assertIn("def factorial", cleaned)

    def test_unknown_prompt_fallback(self):
        code = self.provider.generate_code("Custom binary search tree", SYSTEM_PROMPT)
        cleaned = clean_llm_code_response(code)
        valid, msg = validate_python_syntax(cleaned)
        self.assertTrue(valid, f"Syntax error: {msg}")


class TestUtils(unittest.TestCase):

    def test_prompt_to_filename(self):
        filename = prompt_to_filename("Write a function to check if a number is prime")
        self.assertTrue(filename.endswith(".py"))
        self.assertIn("prime", filename)

    def test_validate_python_syntax_valid(self):
        valid, msg = validate_python_syntax("x = 10\ny = 20\nz = x + y")
        self.assertTrue(valid)

    def test_validate_python_syntax_invalid(self):
        valid, msg = validate_python_syntax("def invalid_func(: pass")
        self.assertFalse(valid)
        self.assertIn("Syntax Error", msg)

    def test_execute_generated_code(self):
        code = "def double(x: int) -> int:\n    return x * 2\n"
        success, msg, scope = execute_generated_code(code)
        self.assertTrue(success)
        self.assertIn("double", scope)
        self.assertEqual(scope["double"](5), 10)


class TestProviderFactory(unittest.TestCase):

    def test_factory_mock(self):
        provider = get_provider("mock")
        self.assertIsInstance(provider, MockProvider)

    def test_factory_groq(self):
        provider = get_provider("groq", api_key="test_key")
        self.assertIsInstance(provider, GroqProvider)
        self.assertEqual(provider.api_key, "test_key")

    def test_factory_ollama(self):
        provider = get_provider("ollama")
        self.assertIsInstance(provider, OllamaProvider)


if __name__ == "__main__":
    unittest.main()
