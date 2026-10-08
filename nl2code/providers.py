"""
LLM Provider Abstractions for Groq, Ollama, and Mock/Offline models.
Uses Python standard library (urllib.request) for zero mandatory dependencies.
"""
import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @abstractmethod
    def generate_code(self, prompt: str, system_prompt: str) -> str:
        """Generates code from a prompt and system prompt."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns readable name of provider."""
        pass


class GroqProvider(LLMProvider):
    """
    Groq Cloud API Provider using OpenAI-compatible chat completions API.
    Offers fast, free tier access to models like Llama 3.3 70B, Llama 3 8B, Gemma 2 9B.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return f"Groq Cloud API ({self.model})"

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                "Groq API Key is missing! Pass --api-key or set GROQ_API_KEY environment variable.\n"
                "You can get a free API key at https://console.groq.com (no credit card needed)."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048
        }

        req = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            raise RuntimeError(f"Groq API HTTP Error {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Groq API: {e.reason}")


class OllamaProvider(LLMProvider):
    """
    Local Ollama LLM Provider running locally (http://localhost:11434).
    Supports llama3.2, qwen2.5-coder, mistral, codellama, etc.
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST).rstrip('/')
        self.api_url = f"{self.host}/api/chat"

    @property
    def provider_name(self) -> str:
        return f"Ollama Local ({self.model} @ {self.host})"

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                if 'message' in res_data and 'content' in res_data['message']:
                    return res_data['message']['content']
                elif 'response' in res_data:
                    return res_data['response']
                else:
                    raise RuntimeError(f"Unexpected Ollama response structure: {res_data}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.host}: {e.reason}.\n"
                f"Make sure Ollama is installed and running (`ollama serve`), or pull model (`ollama pull {self.model}`)."
            )


class MockProvider(LLMProvider):
    """
    Offline/Mock LLM Provider for offline testing, dry-runs, and benchmark suites.
    Provides hand-crafted valid Python code for known benchmark tasks and template code for arbitrary prompts.
    """

    MOCK_RESPONSES = {
        "prime": '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
''',
        "fibonacci": '''def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number efficiently using dynamic programming."""
    if n < 0:
        raise ValueError("Index cannot be negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "factorial": '''def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer with input validation."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n in (0, 1):
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        "sentence": '''def reverse_words(sentence: str) -> str:
    """Reverse words in a sentence while maintaining original word order."""
    words = sentence.split(" ")
    reversed_words = [word[::-1] for word in words]
    return " ".join(reversed_words)
''',
        "palindrome": '''import re

def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome, ignoring spaces, punctuation, and casing."""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return cleaned == cleaned[::-1]
''',
        "vowels": '''from typing import Dict

def count_vowels_consonants(text: str) -> Dict[str, int]:
    """Count the number of vowels and consonants in a string."""
    vowels = set("aeiouAEIOU")
    v_count = 0
    c_count = 0
    for char in text:
        if char.isalpha():
            if char in vowels:
                v_count += 1
            else:
                c_count += 1
    return {"vowels": v_count, "consonants": c_count}
''',
        "flatten": '''from typing import List, Any

def flatten_list(nested: List[Any]) -> List[Any]:
    """Flatten an arbitrarily nested list into a single flat list."""
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat
''',
        "dicts": '''from typing import List, Dict, Union

def merge_dicts_sum_values(dict_list: List[Dict[str, Union[int, float]]]) -> Dict[str, Union[int, float]]:
    """Merge a list of dictionaries by key, summing numerical values for identical keys."""
    merged: Dict[str, Union[int, float]] = {}
    for d in dict_list:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged
''',
        "frequency": '''import re
from typing import Dict

def count_word_frequency(text: str) -> Dict[str, int]:
    """Count frequency of each word in text, removing punctuation and ignoring case."""
    words = re.findall(r'\\b\\w+\\b', text.lower())
    freq: Dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
''',
        "two_sum": '''from typing import List, Tuple

def find_sum_pairs(numbers: List[int], target: int) -> List[Tuple[int, int]]:
    """Find all unique pairs of integers in a list that sum to a target value."""
    seen = set()
    pairs = set()
    for num in numbers:
        complement = target - num
        if complement in seen:
            pair = tuple(sorted((num, complement)))
            pairs.add(pair)
        seen.add(num)
    return list(pairs)
'''
    }

    @property
    def provider_name(self) -> str:
        return "Offline Mock LLM Simulator"

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        prompt_lower = prompt.lower()
        for key, code in self.MOCK_RESPONSES.items():
            if key in prompt_lower:
                # Add fence wrappers occasionally to test fence cleaner robustness!
                return f"```python\n{code}\n```"

        # Fallback template code generator for unrecognized prompts
        slug = "".join(c if c.isalnum() else "_" for c in prompt_lower[:20])
        return f"""def solution_{slug}():
    \"\"\"Generated solution for prompt: {prompt}\"\"\"
    # Auto-generated placeholder logic
    return True
"""


def get_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to initialize requested provider or auto-detect best available.
    """
    p_name = (provider_name or "").strip().lower()

    if p_name == "groq":
        return GroqProvider(api_key=api_key, model=model)
    elif p_name == "ollama":
        return OllamaProvider(model=model)
    elif p_name == "mock":
        return MockProvider()
    
    # Auto-detection mode:
    # 1. If GROQ_API_KEY is present or api_key passed -> Groq
    if api_key or os.environ.get("GROQ_API_KEY"):
        return GroqProvider(api_key=api_key, model=model)
    
    # 2. Otherwise default to Groq (which will prompt for key if missing) or Mock if offline testing
    return GroqProvider(api_key=api_key, model=model)
