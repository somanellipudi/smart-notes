"""
LLM Provider abstraction supporting multiple backends.

Supports:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Ollama (Local models like llama2, mistral, neural-chat)
"""

import logging
import json
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call LLM and return response text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-3.5-turbo)
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.available = True
            logger.info(f"✓ OpenAI provider initialized (model: {model})")
        except Exception as e:
            logger.error(f"✗ OpenAI provider failed to initialize: {e}")
            self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call OpenAI API."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama server URL
            model: Model name (mistral, llama2, neural-chat, etc.)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.available = self._check_availability()
        
        if self.available:
            logger.info(f"✓ Ollama provider initialized (model: {model}, url: {base_url})")
        else:
            logger.warning(f"✗ Ollama provider unavailable at {base_url}")
    
    def is_available(self) -> bool:
        return self.available
    
    def _check_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call Ollama API."""
        try:
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                else:
                    prompt += f"{content}\n"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }
            
            if max_tokens:
                payload["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # Long timeout for local processing
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {}
    
    @staticmethod
    def create_provider(
        provider_type: str,
        api_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMProvider:
        """
        Create LLM provider.
        
        Args:
            provider_type: "openai" or "ollama"
            api_key: OpenAI API key (for OpenAI)
            ollama_url: Ollama base URL (for Ollama)
            model: Model name
        
        Returns:
            LLMProvider instance
        """
        if provider_type == "openai":
            if not api_key:
                raise ValueError("api_key required for OpenAI provider")
            model = model or "gpt-4"
            return OpenAIProvider(api_key=api_key, model=model)
        
        elif provider_type == "ollama":
            ollama_url = ollama_url or "http://localhost:11434"
            model = model or "mistral"
            return OllamaProvider(base_url=ollama_url, model=model)
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers(
        openai_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434"
    ) -> Dict[str, bool]:
        """
        Check which providers are available.
        
        Returns:
            Dictionary with provider names and availability status
        """
        providers = {}
        
        # Check OpenAI
        if openai_api_key:
            try:
                provider = OpenAIProvider(api_key=openai_api_key)
                providers["OpenAI (GPT-4)"] = provider.is_available()
            except:
                providers["OpenAI (GPT-4)"] = False
        
        # Check Ollama
        try:
            provider = OllamaProvider(base_url=ollama_url)
            providers[f"Local LLM - Ollama"] = provider.is_available()
        except:
            providers[f"Local LLM - Ollama"] = False
        
        return providers
