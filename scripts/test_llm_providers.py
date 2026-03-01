#!/usr/bin/env python
"""Quick test of all three LLM providers to verify configuration."""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

def test_providers():
    print("=" * 70)
    print("LLM Provider Configuration Check")
    print("=" * 70)
    print()
    
    # Test OpenAI/GPT-4o
    print("1. OpenAI (GPT-4o)")
    print("   " + "-" * 60)
    if config.OPENAI_API_KEY:
        print(f"   ✓ OPENAI_API_KEY present (length: {len(config.OPENAI_API_KEY)})")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            print(f"   ✓ OpenAI SDK available")
        except Exception as e:
            print(f"   ✗ OpenAI SDK error: {e}")
    else:
        print("   ✗ OPENAI_API_KEY missing")
    print()
    
    # Test Anthropic/Claude
    print("2. Anthropic (Claude)")
    print("   " + "-" * 60)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        print(f"   ✓ ANTHROPIC_API_KEY present (length: {len(anthropic_key)})")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            print(f"   ✓ Anthropic SDK available")
            print(f"   Model: claude-sonnet-4-20250514 (configured in .env)")
        except Exception as e:
            print(f"   ✗ Anthropic SDK error: {e}")
    else:
        print("   ✗ ANTHROPIC_API_KEY missing in environment")
    print()
    
    # Test Ollama/Llama
    print("3. Ollama (Llama)")
    print("   " + "-" * 60)
    try:
        import requests
        url = config.OLLAMA_URL.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            print(f"   ✓ Ollama available at {config.OLLAMA_URL}")
            print(f"   ✓ Available models: {', '.join(models[:3]) if models else 'none'}")
            if "llama2" in models or any("llama" in m.lower() for m in models):
                print(f"   ✓ Llama model available")
            else:
                print(f"   ✗ No Llama model found (available: {models})")
        else:
            print(f"   ✗ Ollama HTTP {r.status_code}")
    except Exception as e:
        print(f"   ✗ Ollama unavailable: {e}")
    print()
    
    print("=" * 70)
    print("Next: Run comparison with: python scripts/run_llm_baseline_comparison.py --max-examples 5")
    print("=" * 70)

if __name__ == "__main__":
    test_providers()
