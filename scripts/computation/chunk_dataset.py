"""
Split large benchmark dataset into chunks for parallel processing.
"""
import json
import argparse
from pathlib import Path


def chunk_dataset(input_path: str, output_dir: str, chunk_size: int):
    """Split JSONL dataset into chunks."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total = len(lines)
    num_chunks = (total + chunk_size - 1) // chunk_size
    
    print(f"Total claims: {total}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")
    
    # Write chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        chunk_lines = lines[start_idx:end_idx]
        
        chunk_path = output_dir / f"chunk_{i+1}_of_{num_chunks}.jsonl"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chunk_lines))
        
        print(f"  Chunk {i+1}/{num_chunks}: {len(chunk_lines)} claims -> {chunk_path.name}")
    
    print(f"\nChunks saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory for chunks")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Claims per chunk")
    
    args = parser.parse_args()
    chunk_dataset(args.input, args.output_dir, args.chunk_size)
