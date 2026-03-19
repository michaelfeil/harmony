"""
Reproduction script for memory issue in openai_harmony.

The issue: StreamableParser.state property calls state_data which does
json.loads(self._inner.state) for EVERY token processed. This causes
excessive memory allocations when processing many tokens.

Memory profile shows:
- raw_decode: 1.827 GB
- state_data: 645 MB
- process: 865,894 allocations
"""

import gc
import json
import sys
import tracemalloc
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Role,
    StreamableParser,
)

# Monkey-patch json.loads to count calls
_original_loads = json.loads
loads_count = 0


def _counting_loads(*args, **kwargs):
    global loads_count
    loads_count += 1
    return _original_loads(*args, **kwargs)


json.loads = _counting_loads


def simulate_streaming_with_state_access(num_tokens: int = 10000):
    """Simulate the pattern from harmony.py that causes memory issues.

    This mimics the code in harmony.py:147-158 where parser.state is
    accessed for every token in the debug logging.
    """
    global loads_count
    loads_count = 0

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    parser = StreamableParser(encoding, role=Role.ASSISTANT)

    # Get some real tokens to process
    test_text = "Hello, this is a test message. " * 100
    tokens = encoding.encode(test_text, allowed_special="all")

    # Cycle through tokens if we need more
    while len(tokens) < num_tokens:
        tokens = tokens + tokens
    tokens = tokens[:num_tokens]

    print(f"Processing {len(tokens)} tokens with state access on each token...")

    for i, token_id in enumerate(tokens):
        parser.process(token_id)
        # This is the problematic pattern - accessing parser.state
        # which internally calls json.loads(self._inner.state)
        _ = parser.state  # <-- This causes the memory issue!
        _ = parser.current_channel
        _ = parser.last_content_delta

    print(f"Done. Processed {len(tokens)} tokens.")
    print(f"json.loads calls: {loads_count}")
    return loads_count


def simulate_streaming_without_state_access(num_tokens: int = 10000):
    """Same as above but without accessing parser.state."""
    global loads_count
    loads_count = 0

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    parser = StreamableParser(encoding, role=Role.ASSISTANT)

    test_text = "Hello, this is a test message. " * 100
    tokens = encoding.encode(test_text, allowed_special="all")

    while len(tokens) < num_tokens:
        tokens = tokens + tokens
    tokens = tokens[:num_tokens]

    print(f"Processing {len(tokens)} tokens WITHOUT state access...")

    for i, token_id in enumerate(tokens):
        parser.process(token_id)
        # Only access properties that don't do JSON parsing
        _ = parser.current_channel
        _ = parser.last_content_delta

    print(f"Done. Processed {len(tokens)} tokens.")
    print(f"json.loads calls: {loads_count}")
    return loads_count


def measure_memory(func, *args):
    """Measure memory usage of a function."""
    gc.collect()
    tracemalloc.start()

    result = func(*args)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    return current, peak, result


if __name__ == "__main__":
    num_tokens = 100000

    print("=" * 70)
    print("Memory usage comparison for openai_harmony StreamableParser")
    print("=" * 70)

    # Test 1: With state access (problematic)
    print(f"\n1. WITH parser.state access (problematic pattern):")
    current1, peak1, loads1 = measure_memory(
        simulate_streaming_with_state_access, num_tokens
    )
    print(f"   Current:  {current1 / 1024 / 1024:.2f} MB")
    print(f"   Peak:     {peak1 / 1024 / 1024:.2f} MB")
    print(f"   json.loads calls: {loads1}")

    # Test 2: Without state access
    print(f"\n2. WITHOUT parser.state access:")
    current2, peak2, loads2 = measure_memory(
        simulate_streaming_without_state_access, num_tokens
    )
    print(f"   Current:  {current2 / 1024 / 1024:.2f} MB")
    print(f"   Peak:     {peak2 / 1024 / 1024:.2f} MB")
    print(f"   json.loads calls: {loads2}")

    print(f"\n" + "=" * 70)
    print(f"Memory difference (peak): {(peak1 - peak2) / 1024 / 1024:.2f} MB")
    print(f"Ratio: {peak1 / peak2:.2f}x more memory with state access")
    print(f"Extra json.loads calls: {loads1 - loads2}")
    print(f"  (each extra call = json.loads of parser state per token)")
    print("=" * 70)
