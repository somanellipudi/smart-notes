#!/usr/bin/env python3
"""
Simple debugger to test Claude API connection and functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_claude():
    """Test Claude API connection and basic functionality."""
    
    print("\n" + "="*80)
    print("CLAUDE API DEBUGGER")
    print("="*80)
    
    # Step 1: Check environment variables
    print("\n[STEP 1] Checking environment variables...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL_PRIMARY")
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env")
        return False
    
    if not model:
        print("❌ ANTHROPIC_MODEL_PRIMARY not found in .env")
        return False
    
    key_preview = api_key[:20] + "..." + api_key[-10:] if len(api_key) > 30 else api_key
    print(f"✅ ANTHROPIC_API_KEY: {key_preview}")
    print(f"✅ ANTHROPIC_MODEL_PRIMARY: {model}")
    
    # Step 2: Import and initialize client
    print("\n[STEP 2] Initializing Anthropic client...")
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized successfully")
    except ImportError as e:
        print(f"❌ Failed to import Anthropic: {e}")
        print("   Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Step 3: Test simple API call
    print("\n[STEP 3] Testing simple API call...")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, Claude is working!' and nothing else."
                }
            ]
        )
        
        result_text = response.content[0].text
        print(f"✅ API call successful!")
        print(f"   Response: {result_text}")
        print(f"   Stop reason: {response.stop_reason}")
        print(f"   Input tokens: {response.usage.input_tokens}")
        print(f"   Output tokens: {response.usage.output_tokens}")
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    # Step 4: Test claim verification task
    print("\n[STEP 4] Testing claim verification...")
    
    test_claim = "Python is a programming language."
    test_source = "Python is a high-level, interpreted programming language known for its simplicity and readability."
    
    try:
        prompt = f"""Verify if the following claim is supported by the source text.

Claim: {test_claim}

Source: {test_source}

Respond ONLY with one word: VERIFIED, REJECTED, or LOW_CONFIDENCE"""
        
        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        verification = response.content[0].text.strip()
        print(f"✅ Claim verification successful!")
        print(f"   Claim: {test_claim}")
        print(f"   Source: {test_source}")
        print(f"   Result: {verification}")
        
    except Exception as e:
        print(f"❌ Claim verification failed: {e}")
        return False
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ All tests passed! Claude is working correctly.")
    print(f"   Model: {model}")
    print(f"   Ready to run: python scripts/run_llm_baseline_comparison.py")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_claude()
    sys.exit(0 if success else 1)
