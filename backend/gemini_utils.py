"""
backend/gemini_utils.py — Enhanced Gemini Integration
==================================================
Centralized Gemini utilities with robust parsing, error handling,
and retry logic for the NeuroLearn language learning platform.

Features:
- Unified model management
- Robust JSON parsing with multiple fallback strategies
- Automatic retry with exponential backoff
- Structured error logging
- Response validation and sanitization
- Rate limiting awareness
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class GeminiConfig:
    """Configuration for Gemini models."""
    api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    
    # Safety settings
    safety_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    
    # Retry settings
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 16.0
    
    # Rate limiting
    requests_per_minute: int = 60


class GeminiManager:
    """Centralized Gemini model management with enhanced features."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        
        # Rate limiting
        self._request_times: List[float] = []
        
        # Model cache
        self._models: Dict[str, genai.GenerativeModel] = {}
        
    def _check_rate_limit(self):
        """Implement simple rate limiting."""
        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self._request_times.append(now)
    
    def get_model(self, temperature: Optional[float] = None) -> genai.GenerativeModel:
        """Get or create a model with specified temperature."""
        temp = temperature or self.config.temperature
        cache_key = f"{self.config.model_name}_{temp:.2f}"
        
        if cache_key not in self._models:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: self.config.safety_threshold,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.config.safety_threshold,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: self.config.safety_threshold,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: self.config.safety_threshold,
            }
            
            self._models[cache_key] = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=self.config.max_output_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                ),
                safety_settings=safety_settings,
            )
        
        return self._models[cache_key]


class JSONParser:
    """Robust JSON parsing with multiple fallback strategies."""
    
    @staticmethod
    def extract_json_from_text(text: str) -> str:
        """Extract JSON from text using multiple strategies."""
        text = text.strip()
        
        # Strategy 1: Look for JSON blocks with code fences
        if "```" in text:
            for part in text.split("```"):
                cleaned = part.lstrip("json").lstrip("JSON").strip()
                if JSONParser._is_valid_json_start(cleaned):
                    return JSONParser._extract_complete_json(cleaned)
        
        # Strategy 2: Find first { and last }
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start < end:
                candidate = text[start:end+1]
                if JSONParser._is_valid_json(candidate):
                    return candidate
        
        # Strategy 3: Find first [ and last ]
        if "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]")
            if start < end:
                candidate = text[start:end+1]
                if JSONParser._is_valid_json(candidate):
                    return candidate
        
        # Strategy 4: Try to fix common JSON issues
        return JSONParser._fix_common_issues(text)
    
    @staticmethod
    def _is_valid_json_start(text: str) -> bool:
        """Check if text starts with valid JSON."""
        return text.startswith("{") or text.startswith("[")
    
    @staticmethod
    def _extract_complete_json(text: str) -> str:
        """Extract complete JSON object or array from text."""
        if text.startswith("{"):
            return JSONParser._extract_balanced(text, "{", "}")
        elif text.startswith("["):
            return JSONParser._extract_balanced(text, "[", "]")
        return text
    
    @staticmethod
    def _extract_balanced(text: str, open_char: str, close_char: str) -> str:
        """Extract balanced brackets/braces."""
        count = 0
        for i, char in enumerate(text):
            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return text[:i+1]
        return text
    
    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    @staticmethod
    def _fix_common_issues(text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix quotes
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        
        # Remove comments
        text = re.sub(r'//.*?\n', '\n', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Try to extract JSON again
        if JSONParser._is_valid_json_start(text):
            return JSONParser._extract_complete_json(text)
        
        return text


class GeminiClient:
    """Enhanced Gemini client with robust error handling and parsing."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.manager = GeminiManager(config)
        self.parser = JSONParser()
    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        expected_schema: Optional[Dict] = None,
        validate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Generate JSON response from Gemini with robust parsing and validation.
        
        Args:
            prompt: The prompt to send to Gemini
            temperature: Optional temperature override
            expected_schema: Optional schema for response validation
            validate_response: Whether to validate the response structure
            
        Returns:
            Parsed JSON response
            
        Raises:
            ValueError: If response cannot be parsed or validated
            RuntimeError: If all retry attempts fail
        """
        self.manager._check_rate_limit()
        model = self.manager.get_model(temperature)
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Gemini request attempt {attempt + 1}/{self.config.max_retries}")
                
                # Generate response
                response = model.generate_content(prompt)
                text = response.text or ""
                
                if not text.strip():
                    raise ValueError("Empty response from Gemini")
                
                # Extract and parse JSON
                json_text = self.parser.extract_json_from_text(text)
                parsed = json.loads(json_text)
                
                # Validate against schema if provided
                if validate_response and expected_schema:
                    self._validate_response(parsed, expected_schema)
                
                logger.debug(f"Successfully parsed JSON response on attempt {attempt + 1}")
                return parsed
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing failed: {e}"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")
                
            except Exception as e:
                last_error = f"Gemini API error: {e}"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")
            
            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                backoff = min(
                    self.config.initial_backoff * (2 ** attempt),
                    self.config.max_backoff
                )
                logger.debug(f"Retrying in {backoff:.2f}s")
                time.sleep(backoff)
        
        raise RuntimeError(f"All retry attempts failed. Last error: {last_error}")
    
    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """Generate text response from Gemini."""
        self.manager._check_rate_limit()
        model = self.manager.get_model(temperature)
        
        for attempt in range(self.config.max_retries):
            try:
                response = model.generate_content(prompt)
                text = response.text or ""
                
                if not text.strip():
                    raise ValueError("Empty response from Gemini")
                
                return text.strip()
                
            except Exception as e:
                logger.warning(f"Text generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    backoff = min(
                        self.config.initial_backoff * (2 ** attempt),
                        self.config.max_backoff
                    )
                    time.sleep(backoff)
        
        raise RuntimeError(f"Text generation failed after {self.config.max_retries} attempts")
    
    def _validate_response(self, response: Dict, schema: Dict):
        """Validate response against expected schema."""
        # Basic validation - can be extended with jsonschema
        for key, expected_type in schema.items():
            if key not in response:
                raise ValueError(f"Missing required key: {key}")
            
            if not isinstance(response[key], expected_type):
                raise ValueError(f"Key {key} should be {expected_type.__name__}, got {type(response[key]).__name__}")


# ── Global Client Instance ─────────────────────────────────────────────────────

# Default configuration
DEFAULT_CONFIG = GeminiConfig(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.7,
    max_retries=3,
    requests_per_minute=60,
)

# Global client instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client(config: Optional[GeminiConfig] = None) -> GeminiClient:
    """Get or create the global Gemini client."""
    global _gemini_client
    
    if _gemini_client is None:
        _gemini_client = GeminiClient(config or DEFAULT_CONFIG)
    
    return _gemini_client


# ── Convenience Functions ─────────────────────────────────────────────────────

def call_gemini_json(
    prompt: str,
    temperature: Optional[float] = None,
    expected_schema: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convenience function for JSON generation."""
    client = get_gemini_client()
    return client.generate_json(prompt, temperature, expected_schema)

def call_gemini_text(
    prompt: str,
    temperature: Optional[float] = None
) -> str:
    """Convenience function for text generation."""
    client = get_gemini_client()
    return client.generate_text(prompt, temperature)


# ── Model Factory Functions ───────────────────────────────────────────────────

def create_story_model() -> GeminiClient:
    """Create a client optimized for story generation."""
    config = GeminiConfig(
        api_key=DEFAULT_CONFIG.api_key,
        model_name=DEFAULT_CONFIG.model_name,
        temperature=0.9,  # Higher creativity for stories
        max_output_tokens=3072,
    )
    return GeminiClient(config)

def create_question_model() -> GeminiClient:
    """Create a client optimized for question generation."""
    config = GeminiConfig(
        api_key=DEFAULT_CONFIG.api_key,
        model_name=DEFAULT_CONFIG.model_name,
        temperature=0.8,  # Balanced creativity
        max_output_tokens=2048,
    )
    return GeminiClient(config)

def create_feedback_model() -> GeminiClient:
    """Create a client optimized for feedback generation."""
    config = GeminiConfig(
        api_key=DEFAULT_CONFIG.api_key,
        model_name=DEFAULT_CONFIG.model_name,
        temperature=0.85,  # Warm and encouraging tone
        max_output_tokens=1024,
    )
    return GeminiClient(config)

def create_reasoning_model() -> GeminiClient:
    """Create a client optimized for reasoning tasks."""
    config = GeminiConfig(
        api_key=DEFAULT_CONFIG.api_key,
        model_name=DEFAULT_CONFIG.model_name,
        temperature=0.2,  # Low temperature for consistent reasoning
        max_output_tokens=1536,
    )
    return GeminiClient(config)
