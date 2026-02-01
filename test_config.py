#!/usr/bin/env python
"""Quick test to verify configuration is loading correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test config loading
    from config import (
        ENVIRONMENT,
        DEBUG,
        LOG_LEVEL,
        OPENAI_API_KEY,
        LLM_MODEL,
        WHISPER_MODEL_SIZE,
        is_development,
        is_production,
    )
    
    print("✓ Configuration loaded successfully!")
    print(f"  ENVIRONMENT: {ENVIRONMENT}")
    print(f"  DEBUG: {DEBUG}")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  LLM_MODEL: {LLM_MODEL}")
    print(f"  WHISPER_MODEL_SIZE: {WHISPER_MODEL_SIZE}")
    print(f"  OPENAI_API_KEY: {'*' * 10 if OPENAI_API_KEY else '<not set>'}")
    print(f"  is_development(): {is_development()}")
    print(f"  is_production(): {is_production()}")
    print("\n✓ All configuration values loaded correctly!")
    
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
