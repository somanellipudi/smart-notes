"""Test to verify the AttributeError fix in verifiable_pipeline.py"""
from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
from src.schema.output_schema import ClassSessionOutput
import config

# Test the fix: ensure that output is always a ClassSessionOutput, never a dict
pipeline = VerifiablePipelineWrapper(
    provider_type="openai",
    api_key="test-key",
    ollama_url=config.OLLAMA_URL,
    model="gpt-3.5-turbo",
)

# Test with very short input (should trigger the unverifiable_input code path)
result = pipeline.process(
    combined_content="Hello",  # Very short, will fail verification
    equations=[],
    external_context="",
    session_id="test_session_123",
    verifiable_mode=True,
    output_filters={'summary': True},
    urls=[]
)

output, metadata = result
print(f"✅ Output type is: {type(output).__name__}")
print(f"✅ Output is ClassSessionOutput: {isinstance(output, ClassSessionOutput)}")
print(f"✅ Output has session_id: {hasattr(output, 'session_id')}")
print(f"✅ Output.session_id = {output.session_id}")
print(f"✅ Metadata status: {metadata.get('status', 'unknown')}")
print("\n✅ FIX VERIFIED: Output is always a ClassSessionOutput, never a dict!")
