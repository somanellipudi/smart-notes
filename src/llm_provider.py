"""
LLM Provider abstraction supporting multiple backends.

Supports:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Ollama (Local models like llama2, mistral, neural-chat)
"""

import logging
import json
import os
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

try:
    import config
    _ENABLE_LANGCHAIN_TRACING = config.ENABLE_LANGCHAIN_TRACING
    _LANGCHAIN_API_KEY = config.LANGCHAIN_API_KEY
    _LANGCHAIN_PROJECT_ID = config.LANGCHAIN_PROJECT_ID
except Exception:
    _ENABLE_LANGCHAIN_TRACING = False
    _LANGCHAIN_API_KEY = ""
    _LANGCHAIN_PROJECT_ID = ""

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

    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convenience wrapper for text-only prompts."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-3.5-turbo)
        """
        self.model = model
        self._langchain_client = None
        self._use_langchain = False
        try:
            # Optional LangChain integration for tracing/monitoring
            if _ENABLE_LANGCHAIN_TRACING and _LANGCHAIN_API_KEY:
                os.environ.setdefault("LANGCHAIN_API_KEY", _LANGCHAIN_API_KEY)
                os.environ.setdefault("LANGCHAIN_PROJECT", _LANGCHAIN_PROJECT_ID or "smart-notes")
                os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
                from langchain_openai import ChatOpenAI
                self._langchain_client = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    api_key=api_key
                )
                self._use_langchain = True
                self.available = True
                logger.info(f"✓ OpenAI provider initialized with LangChain (model: {model})")
                return

            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
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
            if self._use_langchain and self._langchain_client is not None:
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                lc_messages = []
                for message in messages:
                    role = message.get("role")
                    content = message.get("content", "")
                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    else:
                        lc_messages.append(HumanMessage(content=content))

                bind_kwargs: Dict[str, Any] = {}
                if temperature is not None:
                    bind_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    bind_kwargs["max_tokens"] = max_tokens
                if response_format is not None:
                    bind_kwargs["response_format"] = response_format

                client = self._langchain_client.bind(**bind_kwargs) if bind_kwargs else self._langchain_client
                try:
                    response = client.invoke(lc_messages)
                    return response.content
                except Exception as lc_err:
                    if response_format and "response_format" in str(lc_err):
                        logger.warning("LangChain response_format not supported. Retrying without response_format.")
                        client = self._langchain_client
                        response = client.invoke(lc_messages)
                        return response.content
                    raise

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
            error_text = str(e)
            if response_format and "response_format" in error_text:
                logger.warning("OpenAI response_format not supported for this model. Retrying without response_format.")
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if max_tokens:
                        kwargs["max_tokens"] = max_tokens
                    response = self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                except Exception as retry_err:
                    logger.error(f"OpenAI API error after retry: {retry_err}")
                    raise

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
        """Call Ollama API using chat endpoint."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            # Note: response_format (JSON mode) is not supported by all Ollama models
            # If needed, it should be added to the system prompt instead
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300  # Long timeout for local processing
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
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
