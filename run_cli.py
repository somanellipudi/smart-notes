"""
CLI runner for Smart Notes with URL ingestion support.

Usage:
    python -m smart_notes.run --input notes.txt --urls "https://youtu.be/..." "https://example.com/article"
    python run_cli.py --input notes.txt --urls "https://youtu.be/..." --verifiable
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
from src.preprocessing.text_processing import preprocess_classroom_content

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Notes CLI with URL ingestion support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process notes with YouTube video
  python run_cli.py --input notes.txt --urls "https://www.youtube.com/watch?v=abc123"
  
  # Process with multiple URLs
  python run_cli.py --input notes.txt --urls "https://youtu.be/abc" "https://example.com/article"
  
  # Enable debug mode
  python run_cli.py --input notes.txt --urls "https://youtu.be/abc" --debug --relaxed
        """
    )
    
    # Input arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input notes file (text)'
    )
    
    parser.add_argument(
        '--urls', '-u',
        type=str,
        nargs='+',
        help='URLs to ingest as evidence sources (YouTube videos or articles)'
    )
    
    parser.add_argument(
        '--external',
        type=str,
        help='Path to external context file (optional)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file path (default: outputs/sessions/session_<timestamp>.json)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--verifiable',
        action='store_true',
        default=True,
        help='Enable verifiable mode (default: True)'
    )
    
    parser.add_argument(
        '--standard',
        action='store_true',
        help='Use standard mode instead of verifiable mode'
    )
    
    # Debug arguments
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug verification logging'
    )
    
    parser.add_argument(
        '--relaxed',
        action='store_true',
        help='Use relaxed verification thresholds'
    )
    
    # LLM arguments
    parser.add_argument(
        '--model',
        type=str,
        default=config.LLM_MODEL,
        help=f'LLM model to use (default: {config.LLM_MODEL})'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'ollama'],
        default='openai',
        help='LLM provider (default: openai)'
    )
    
    args = parser.parse_args()
    
    # Set debug flags
    if args.debug:
        config.DEBUG_VERIFICATION = True
        config.SAVE_DEBUG_REPORT = True
        logger.info("Debug verification enabled")
    
    if args.relaxed:
        config.RELAXED_VERIFICATION_MODE = True
        logger.info("Relaxed verification mode enabled")
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info(f"Reading input from: {args.input}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Read external context if provided
    external_context = ""
    if args.external:
        external_path = Path(args.external)
        if not external_path.exists():
            logger.warning(f"External context file not found: {args.external}")
        else:
            logger.info(f"Reading external context from: {args.external}")
            with open(external_path, 'r', encoding='utf-8') as f:
                external_context = f.read()
    
    # Preprocess content
    logger.info("Preprocessing content...")
    combined_content = preprocess_classroom_content(
        transcript=content,
        notes="",
        handwriting=""
    )
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline: {args.provider} / {args.model}")
    pipeline = VerifiablePipelineWrapper(
        model=args.model,
        provider_type=args.provider,
        api_key=config.OPENAI_API_KEY if args.provider == 'openai' else None,
        ollama_url=config.OLLAMA_URL if args.provider == 'ollama' else None
    )
    
    # Determine verifiable mode
    verifiable_mode = args.verifiable and not args.standard
    
    # Process
    logger.info(f"Processing with verifiable_mode={verifiable_mode}...")
    if args.urls:
        logger.info(f"URLs to ingest: {len(args.urls)}")
        for url in args.urls:
            logger.info(f"  - {url}")
    
    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        output, metadata = pipeline.process(
            combined_content=combined_content,
            equations=[],
            external_context=external_context,
            session_id=session_id,
            verifiable_mode=verifiable_mode,
            urls=args.urls or []
        )
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = config.SESSIONS_DIR / f"session_{session_id}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path),
            "urls": args.urls or [],
            "verifiable_mode": verifiable_mode,
            "output": output.to_dict() if hasattr(output, 'to_dict') else output,
        }
        
        if metadata:
            # Add metadata but filter out non-serializable objects
            output_data["metadata"] = {
                "verifiable_mode": metadata.get("verifiable_mode"),
                "domain_profile": metadata.get("domain_profile"),
                "processing_time": metadata.get("processing_time"),
                "auto_relaxed_retry": metadata.get("auto_relaxed_retry"),
                "url_ingestion_summary": metadata.get("url_ingestion_summary"),
                "metrics": metadata.get("metrics"),
                "graph_metrics": metadata.get("graph_metrics_dict"),
                "timings": metadata.get("timings"),
            }
            
            # Add claim counts
            if metadata.get("claim_collection"):
                output_data["metadata"]["total_claims"] = len(metadata["claim_collection"].claims)
            if metadata.get("verified_collection"):
                output_data["metadata"]["verified_claims"] = len(metadata["verified_collection"].claims)
        
        # Save output
        logger.info(f"Saving output to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Processing complete!")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Session ID: {session_id}")
        print(f"Output saved: {output_path}")
        
        if metadata:
            print(f"Processing time: {metadata.get('processing_time', 0):.1f}s")
            
            if metadata.get('url_ingestion_summary'):
                summary = metadata['url_ingestion_summary']
                print(f"\nURL Ingestion:")
                print(f"  Total: {summary.get('total_urls', 0)}")
                print(f"  Successful: {summary.get('successful', 0)}")
                print(f"  Failed: {summary.get('failed', 0)}")
                print(f"  Total chars: {summary.get('total_chars', 0):,}")
            
            if metadata.get('claim_collection'):
                total = len(metadata['claim_collection'].claims)
                verified = len(metadata.get('verified_collection', {}).claims) if metadata.get('verified_collection') else 0
                print(f"\nClaims:")
                print(f"  Total: {total}")
                print(f"  Verified: {verified}")
                print(f"  Verification rate: {verified/total*100:.1f}%" if total > 0 else "  Verification rate: N/A")
            
            if metadata.get('auto_relaxed_retry'):
                print("\n⚠️  Auto-relaxed retry was triggered due to high rejection rate")
        
        print("="*70)
        
        if config.DEBUG_VERIFICATION and config.SAVE_DEBUG_REPORT:
            debug_report_path = config.OUTPUT_DIR / config.DEBUG_REPORT_PATH
            print(f"\n📊 Debug report saved: {debug_report_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
