"""
FineWeb-Edu preprocessing for DumGPT.

Downloads FineWeb-Edu in streaming mode, tokenizes documents using the GPT-2
tokenizer, and saves the token IDs into uint16 binary shards.

Output:
    edu_fineweb10B/
        edu_fineweb_val_000000.bin
        edu_fineweb_train_000001.bin
        edu_fineweb_train_000002.bin
        ...

Run:
    python fineweb_edu.py

For a small test:
    python fineweb_edu.py --max-documents 1000

For Kaggle (recommended initial dataset budget):
    python fineweb_edu.py --max-tokens 750000000 --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"

OUTPUT_DIR = Path("/kaggle/working/edu_fineweb10B")

# Number of tokens stored in one shard.
# 100 million uint16 tokens occupy approximately 200 MB.
SHARD_SIZE = 100_000_000

# Use a limited number of CPU workers to avoid excessive RAM usage.
CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = max(1, min(8, CPU_COUNT // 2))
POOL_CHUNKSIZE = 16

# Optional limits. None means process the complete selected dataset.
MAX_DOCUMENTS: int | None = None
MAX_TOKENS: int | None = None

# Never overwrite existing shards unless explicitly enabled.
OVERWRITE_EXISTING_SHARDS = False


# =============================================================================
# TOKENIZER
# =============================================================================

ENCODER = tiktoken.get_encoding("gpt2")
EOT_TOKEN = ENCODER.eot_token


def tokenize_document(document: dict) -> np.ndarray:
    """Convert one FineWeb document into GPT-2 token IDs."""
    text = document.get("text", "")

    if not isinstance(text, str):
        text = str(text)

    # Add end-of-text token before every document.
    token_ids = [EOT_TOKEN]
    token_ids.extend(ENCODER.encode_ordinary(text))

    # GPT-2 vocabulary IDs fit safely inside uint16.
    return np.asarray(token_ids, dtype=np.uint16)


# =============================================================================
# SHARD WRITING
# =============================================================================

def get_shard_path(shard_index: int) -> Path:
    """Validation is shard 0; all remaining shards are training data."""
    split = "val" if shard_index == 0 else "train"
    return OUTPUT_DIR / f"edu_fineweb_{split}_{shard_index:06d}.bin"


def write_shard(path: Path, tokens: np.ndarray) -> None:
    """Write uint16 tokens directly to disk."""
    with path.open("wb") as file:
        tokens.tofile(file)


def check_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(OUTPUT_DIR.glob("*.bin"))

    if existing_shards and not OVERWRITE_EXISTING_SHARDS:
        examples = "\n".join(f"  - {p.name}" for p in existing_shards[:5])
        raise FileExistsError(
            f"Found existing shards in {OUTPUT_DIR}:\n{examples}\n"
            "Move them, choose another output directory, or enable overwrite."
        )

    if OVERWRITE_EXISTING_SHARDS:
        for path in existing_shards:
            path.unlink()


def write_token_shards(
    tokenized_documents: Iterator[np.ndarray],
    max_tokens: int | None = None,
) -> tuple[int, int]:
    """
    Pack tokenized documents into fixed-size shards.

    A document can cross a shard boundary. We copy the remaining part into
    the next shard so that no tokens are dropped.
    """
    shard_buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    tokens_in_buffer = 0
    shard_index = 0
    total_tokens_written = 0

    progress = tqdm(unit="tokens", unit_scale=True, desc="Shard 0 (validation)")

    def flush_shard(token_count: int, index: int) -> None:
        path = get_shard_path(index)
        write_shard(path, shard_buffer[:token_count])
        tqdm.write(
            f"Saved {path.name}: {token_count:,} tokens "
            f"({path.stat().st_size / (1024**2):.1f} MiB)"
        )

    try:
        for document_tokens in tokenized_documents:
            offset = 0

            while offset < len(document_tokens):
                # Stop once the requested total token budget is reached.
                if max_tokens is not None and total_tokens_written >= max_tokens:
                    break

                available_space = SHARD_SIZE - tokens_in_buffer
                remaining_document = len(document_tokens) - offset
                count = min(available_space, remaining_document)

                if max_tokens is not None:
                    count = min(count, max_tokens - total_tokens_written)

                shard_buffer[tokens_in_buffer:tokens_in_buffer + count] = (
                    document_tokens[offset:offset + count]
                )

                tokens_in_buffer += count
                offset += count
                total_tokens_written += count
                progress.update(count)

                if tokens_in_buffer == SHARD_SIZE:
                    flush_shard(tokens_in_buffer, shard_index)
                    progress.close()

                    shard_index += 1
                    tokens_in_buffer = 0

                    split_label = "training"
                    progress = tqdm(
                        unit="tokens",
                        unit_scale=True,
                        desc=f"Shard {shard_index} ({split_label})",
                    )

            if max_tokens is not None and total_tokens_written >= max_tokens:
                break

        # Save the last partially filled shard.
        if tokens_in_buffer:
            flush_shard(tokens_in_buffer, shard_index)

    finally:
        progress.close()

    shard_count = shard_index + int(tokens_in_buffer > 0)
    return total_tokens_written, shard_count


# =============================================================================
# DATASET DOWNLOAD AND PREPROCESSING
# =============================================================================

def main() -> None:
    global OUTPUT_DIR, SHARD_SIZE, NUM_WORKERS
    global MAX_DOCUMENTS, MAX_TOKENS, OVERWRITE_EXISTING_SHARDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--workers", type=int, default=min(4, NUM_WORKERS))
    parser.add_argument("--max-documents", type=int, default=MAX_DOCUMENTS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    SHARD_SIZE = args.shard_size
    NUM_WORKERS = args.workers
    MAX_DOCUMENTS = args.max_documents
    MAX_TOKENS = args.max_tokens
    OVERWRITE_EXISTING_SHARDS = args.overwrite

    if SHARD_SIZE <= 0 or NUM_WORKERS <= 0:
        raise ValueError("Shard size and worker count must be positive.")

    print("=" * 72)
    print("FineWeb-Edu preprocessing")
    print("=" * 72)
    print(f"Dataset       : {DATASET_NAME}")
    print(f"Configuration : {DATASET_CONFIG}")
    print(f"Split         : {DATASET_SPLIT}")
    print(f"Output        : {OUTPUT_DIR}")
    print(f"Shard size    : {SHARD_SIZE:,} tokens")
    print(f"Workers       : {NUM_WORKERS}")
    print(f"Max documents : {MAX_DOCUMENTS or 'all'}")
    print(f"Max tokens    : {MAX_TOKENS or 'all'}")
    print()

    check_output_directory()

    print("Opening FineWeb-Edu in streaming mode...")
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    if MAX_DOCUMENTS is not None:
        dataset = dataset.take(MAX_DOCUMENTS)

    print("Dataset stream ready. Starting tokenization...\n")

    # Pool creation stays inside main() for Windows multiprocessing safety.
    with mp.Pool(processes=NUM_WORKERS) as pool:
        tokenized_documents = pool.imap(
            tokenize_document,
            dataset,
            chunksize=POOL_CHUNKSIZE,
        )
        total_tokens, shard_count = write_token_shards(
            tokenized_documents,
            max_tokens=MAX_TOKENS,
        )

    total_size = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.bin"))

    print("\n" + "=" * 72)
    print("Preprocessing complete")
    print("=" * 72)
    print(f"Tokens written : {total_tokens:,}")
    print(f"Shards written : {shard_count}")
    print(f"Output folder  : {OUTPUT_DIR.resolve()}")
    print(f"Total size     : {total_size / (1024**3):.2f} GiB")

    if shard_count < 2:
        print("WARNING: At least two shards are recommended: one validation and one training.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
