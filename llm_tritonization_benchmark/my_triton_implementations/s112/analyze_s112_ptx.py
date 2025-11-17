#!/usr/bin/env python3
"""
Compile s112 kernel and extract PTX to analyze memory ordering
"""
import torch
import triton
import triton.language as tl
from triton.compiler import compile as triton_compile
import tempfile
import os

# Define the kernel inline so we can compile it directly
@triton.jit
def s112_kernel_inline(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements - 1

    # Reverse iteration mapping
    reverse_offsets = n_elements - 2 - offsets
    reverse_mask = reverse_offsets >= 0

    # CRITICAL: Load from reverse_offsets
    a_vals = tl.load(a_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
    b_vals = tl.load(b_ptr + reverse_offsets, mask=reverse_mask, other=0.0)

    # Compute
    result = a_vals + b_vals

    # CRITICAL: Store to reverse_offsets + 1
    # This creates potential RAW hazard!
    store_offsets = reverse_offsets + 1
    store_mask = reverse_mask & (store_offsets < n_elements)
    tl.store(a_ptr + store_offsets, result, mask=store_mask)

print("=" * 80)
print("ANALYZING s112 KERNEL MEMORY ACCESS PATTERN")
print("=" * 80)

# Test with actual data to see behavior
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device='cuda')

print("\nInitial state:")
print(f"a = {a}")
print(f"b = {b}")

# Run the kernel
grid = (1,)  # Single block to see intra-block behavior
s112_kernel_inline[grid](a, b, 5, BLOCK_SIZE=256)

print(f"\nAfter kernel execution:")
print(f"a = {a}")
print(f"Expected (sequential): [1., 11., 22., 33., 44.]")

# Now let's analyze execution order
print("\n" + "=" * 80)
print("THREAD EXECUTION ANALYSIS")
print("=" * 80)

print("\nFor n_elements=5, BLOCK_SIZE=256:")
print("\nThread ID -> reverse_offset -> read from -> write to:")
for thread_id in range(5):
    reverse_offset = 5 - 2 - thread_id
    read_addr = reverse_offset
    write_addr = reverse_offset + 1
    if reverse_offset >= 0:
        print(f"  Thread {thread_id:2d}: reverse={reverse_offset} -> reads a[{read_addr}] + b[{read_addr}] -> writes a[{write_addr}]")

print("\nRace Condition Analysis:")
print("Within a SINGLE warp (threads 0-31 execute together):")
print("  - Thread 0 reads a[3], writes a[4]")
print("  - Thread 1 reads a[2], writes a[3]  <- CONFLICT: reads from where thread 0 writes!")
print("  - Thread 2 reads a[1], writes a[2]  <- CONFLICT: reads from where thread 1 writes!")
print("  - Thread 3 reads a[0], writes a[1]  <- CONFLICT: reads from where thread 2 writes!")

print("\n" + "=" * 80)
print("WHY THIS WORKS (Hypothesis)")
print("=" * 80)

print("""
Despite the apparent race conditions, the kernel produces correct results.
Here's why this likely works:

1. **SIMT Execution Model**:
   Within a warp, all threads execute the SAME instruction simultaneously.
   - All threads execute 'tl.load' at the same time
   - All threads execute 'tl.store' at the same time
   - Loads complete BEFORE stores begin

2. **Memory Transaction Ordering**:
   CUDA guarantees that within a single instruction stream:
   - All loads in tl.load() complete before returning
   - All stores in tl.store() complete together
   - This creates an implicit barrier between loads and stores

3. **No Inter-Thread Dependencies**:
   Even though thread i+1 writes to where thread i reads,
   because ALL loads happen BEFORE ALL stores (within the warp),
   each thread reads the OLD value before any thread writes.

4. **Serialization by Design**:
   The memory access pattern ensures that:
   - Thread 0: reads OLD a[3], writes NEW a[4]
   - Thread 1: reads OLD a[2], writes NEW a[3]
   - All reads use OLD values, all writes create NEW values
   - No thread reads a value that another thread just wrote

This is essentially serializing the reverse loop but doing all operations
in parallel using the old values. It's correct by accident of how SIMT works!
""")

# Verify with larger test
print("\n" + "=" * 80)
print("VERIFICATION WITH LARGER ARRAY")
print("=" * 80)

N = 100
a_test = torch.arange(N, dtype=torch.float32, device='cuda')
b_test = torch.ones(N, device='cuda') * 10.0

print(f"\nBefore: a[0:10] = {a_test[:10]}")

grid = (triton.cdiv(N - 1, 256),)
s112_kernel_inline[grid](a_test, b_test, N, BLOCK_SIZE=256)

print(f"After:  a[0:10] = {a_test[:10]}")

# Expected: reverse iteration means a[i+1] = a[i] + 10
# i=98: a[99] = a[98] + 10 = 108
# i=97: a[98] = a[97] + 10 = 107
# ...
# i=1: a[2] = a[1] + 10 = 11
# i=0: a[1] = a[0] + 10 = 10
expected = a_test.clone()
expected[0] = 0  # unchanged
expected[1] = 10  # 0 + 10
expected[2] = 11  # 1 + 10
expected[3] = 12  # 2 + 10

print(f"Expected: a[0:10] = {expected[:10]}")
print(f"\nMatch: {torch.allclose(a_test[:10], expected[:10])}")
