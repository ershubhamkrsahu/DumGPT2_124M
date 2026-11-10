"""
FineWeb-Edu preprocessing script for DumGPT.

Downloads FineWeb-Edu as a stream, tokenizes documents with GPT-2's
tiktoken tokenizer, and writes uint16 binary shards.

Default output:
    edu_fineweb10B/
        edu_fineweb_val_000000.bin
        edu_fineweb_train_000001.bin
        edu_fineweb_train_000002.bin
        ...

Run:
    uv run fineweb_edu.py

Notes:
- The first shard is reserved for validation; remaining shards are training data.
- Streaming avoids downloading/materializing the entire dataset before processing.
- Each token is stored as uint16 (GPT-2 vocabulary IDs fit in this type).
- Existing .bin shards are not overwritten by default.
"""

from __future__ import annotations

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

OUTPUT_DIR = Path(__file__).resolve().parent / "edu_fineweb10B"

# Number of tokens stored in each shard.
# 100 million uint16 tokens occupy approximately 200 MB.
SHARD_SIZE = 100_000_000

# Use at most half the available CPU cores, capped to avoid excessive
# process creation and memory pressure on Windows.
CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = max(1, min(8, CPU_COUNT // 2))

# Pool task batch size. Larger values reduce scheduling overhead, but may
# increase memory use and make progress updates less granular.
POOL_CHUNKSIZE = 16

# Set to an integer to process only that many documents while testing.
# Keep as None to process the complete selected dataset configuration.
MAX_DOCUMENTS: int | None = None

# Safety: refuse to overwrite existing binary shards.
OVERWRITE_EXISTING_SHARDS = False

# =============================================================================
# TOKENIZER
# =============================================================================

# GPT-2 tokenizer. Created at module scope so each spawned worker can access it.
ENCODER = tiktoken.get_encoding("gpt2")
EOT_TOKEN = ENCODER.eot_token

def tokenize_document(document: dict) -> np.ndarray:
    """
    Convert one FineWeb document into GPT-2 token IDs.

    An end-of-text token is prepended to each document, matching the usual
    GPT-2/FineWeb preprocessing convention.
    """
    text = document.get("text", "")

    if not isinstance(text, str):
        text = str(text)

    token_ids = [EOT_TOKEN]
    token_ids.extend(ENCODER.encode_ordinary(text))

    tokens = np.asarray(token_ids, dtype=np.int64)

    # GPT-2 token IDs fit safely in uint16.
    if tokens.size and (tokens.min() < 0 or tokens.max() >= 2**16):
        raise ValueError("A token ID does not fit in uint16.")

    return tokens.astype(np.uint16, copy=False)

# =============================================================================
# SHARD WRITING
# =============================================================================

def get_shard_path(shard_index: int) -> Path:
    """Return the output path for a shard index."""
    split = "val" if shard_index == 0 else "train"
    return OUTPUT_DIR / f"edu_fineweb_{split}_{shard_index:06d}.bin"


def write_shard(path: Path, tokens: np.ndarray) -> None:
    """Write a NumPy uint16 array directly to a binary file."""
    with path.open("wb") as file:
        tokens.tofile(file)


def check_output_directory() -> None:
    """Create the output directory and protect existing shards."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(OUTPUT_DIR.glob("*.bin"))

    if existing_shards and not OVERWRITE_EXISTING_SHARDS:
        examples = "\n".join(f"  - {path.name}" for path in existing_shards[:5])
        more = (
            f"\n  ... and {len(existing_shards) - 5} more"
            if len(existing_shards) > 5
            else ""
        )
        raise FileExistsError(
            f"Found {len(existing_shards)} existing .bin shard(s) in:\n"
            f"{OUTPUT_DIR}\n"
            f"{examples}{more}\n\n"
            "To avoid accidental data loss, this script will not overwrite them. "
            "Move/delete the old shards or set "
            "OVERWRITE_EXISTING_SHARDS = True after checking the directory."
        )


def write_token_shards(tokenized_documents: Iterator[np.ndarray]) -> tuple[int, int]:
    """
    Pack tokenized documents into fixed-size shards.

    Documents may cross shard boundaries. The token stream is split safely
    without dropping or duplicating tokens.
    """
    shard_buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    tokens_in_buffer = 0
    shard_index = 0
    total_tokens_written = 0

    progress = tqdm(
        total=SHARD_SIZE,
        unit="tokens",
        unit_scale=True,
        desc="Shard 0 (validation)",
    )

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

            # A document can be larger than the remaining shard capacity,
            # so keep copying pieces until the complete document is consumed.
            while offset < len(document_tokens):
                available_space = SHARD_SIZE - tokens_in_buffer
                count = min(available_space, len(document_tokens) - offset)

                shard_buffer[
                    tokens_in_buffer : tokens_in_buffer + count
                ] = document_tokens[offset : offset + count]

                tokens_in_buffer += count
                offset += count
                total_tokens_written += count
                progress.update(count)

                if tokens_in_buffer == SHARD_SIZE:
                    flush_shard(tokens_in_buffer, shard_index)
                    progress.close()

                    shard_index += 1
                    tokens_in_buffer = 0

                    split_label = "validation" if shard_index == 0 else "training"
                    progress = tqdm(
                        total=SHARD_SIZE,
                        unit="tokens",
                        unit_scale=True,
                        desc=f"Shard {shard_index} ({split_label})",
                    )

        # Write the final partial shard, if any tokens remain.
        if tokens_in_buffer:
            flush_shard(tokens_in_buffer, shard_index)

    finally:
        progress.close()

    number_of_shards = shard_index + (1 if tokens_in_buffer else 0)
    return total_tokens_written, number_of_shards

# =============================================================================
# DATASET DOWNLOAD AND PREPROCESSING
# =============================================================================

def main() -> None:
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
    print()

    check_output_directory()

    print("Opening the dataset in streaming mode...")
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # Optionally limit the number of documents for a small test run.
    if MAX_DOCUMENTS is not None:
        if MAX_DOCUMENTS <= 0:
            raise ValueError("MAX_DOCUMENTS must be a positive integer or None.")
        dataset = dataset.take(MAX_DOCUMENTS)

    print("Dataset stream ready.")
    print("Starting tokenization and shard writing...\n")

    # On Windows, multiprocessing uses 'spawn'. Keeping pool creation inside
    # main() prevents child processes from rerunning the whole script.
    with mp.Pool(processes=NUM_WORKERS) as pool:
        tokenized_documents = pool.imap(
            tokenize_document,
            dataset,
            chunksize=POOL_CHUNKSIZE,
        )

        total_tokens, shard_count = write_token_shards(tokenized_documents)

    print("\n" + "=" * 72)
    print("Preprocessing complete")
    print("=" * 72)
    print(f"Documents limit : {MAX_DOCUMENTS or 'all'}")
    print(f"Tokens written  : {total_tokens:,}")
    print(f"Shards written  : {shard_count}")
    print(f"Output folder   : {OUTPUT_DIR}")
    print(f"Total size      : {sum(p.stat().st_size for p in OUTPUT_DIR.glob('*.bin')) / (1024**3):.2f} GiB")

if __name__ == "__main__":
    # Required for safe multiprocessing on Windows.
    mp.freeze_support()
    main()