#!/usr/bin/env python3
"""
Demonstrate that s112's correctness relies on SIMT execution model
and show when this pattern breaks
"""
import torch
import triton
import triton.language as tl

# Version 1: s112's actual pattern (WORKS due to implicit load/store ordering)
@triton.jit
def s112_pattern_safe(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    This works because tl.load and tl.store create implicit barriers:
    ALL loads complete BEFORE ANY stores begin
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    reverse_idx = n - 2 - offsets
    mask = reverse_idx >= 0

    # ALL threads load here (old values)
    vals = tl.load(a_ptr + reverse_idx, mask=mask, other=0.0)

    # Implicit barrier here!
    # ALL threads store here (new values)
    tl.store(a_ptr + reverse_idx + 1, vals + 1.0, mask=mask)


# Version 2: Breaking the pattern with intermediate operations
@triton.jit
def s112_pattern_unsafe(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    This MIGHT break if Triton doesn't preserve load/store ordering
    with intermediate operations
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    reverse_idx = n - 2 - offsets
    mask = reverse_idx >= 0

    # Load
    vals = tl.load(a_ptr + reverse_idx, mask=mask, other=0.0)

    # Multiple intermediate operations
    temp1 = vals * 2.0
    temp2 = temp1 / 2.0
    temp3 = temp2 + 1.0
    temp4 = temp3 - 0.0
    result = temp4

    # Store after many operations
    tl.store(a_ptr + reverse_idx + 1, result, mask=mask)


# Version 3: Multiple load/store pairs (WILL break!)
@triton.jit
def s112_pattern_broken(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    This WILL produce incorrect results because we do:
    load -> store -> load -> store
    The second load might see the first store's results!
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    reverse_idx = n - 2 - offsets
    mask = reverse_idx >= 0

    # First load-store pair
    vals1 = tl.load(a_ptr + reverse_idx, mask=mask, other=0.0)
    tl.store(a_ptr + reverse_idx + 1, vals1 + 1.0, mask=mask)

    # Second load-store pair - THIS WILL SEE MODIFIED DATA!
    vals2 = tl.load(a_ptr + reverse_idx, mask=mask, other=0.0)
    tl.store(a_ptr + reverse_idx + 1, vals2 + 1.0, mask=mask)


print("=" * 80)
print("S112 RACE CONDITION ANALYSIS")
print("=" * 80)

N = 10

# Test 1: Safe pattern (s112's actual implementation)
print("\n[TEST 1] Safe pattern (s112's implementation)")
print("-" * 80)
a1 = torch.arange(N, dtype=torch.float32, device='cuda')
print(f"Before: {a1}")
s112_pattern_safe[(1,)](a1, N, BLOCK_SIZE=256)
print(f"After:  {a1}")
print(f"Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + 1 shifted")
print(f"Correct: {torch.equal(a1, torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda'))}")

# Test 2: Unsafe pattern with intermediate ops
print("\n[TEST 2] Pattern with intermediate operations")
print("-" * 80)
a2 = torch.arange(N, dtype=torch.float32, device='cuda')
print(f"Before: {a2}")
s112_pattern_unsafe[(1,)](a2, N, BLOCK_SIZE=256)
print(f"After:  {a2}")
print(f"Still works? {torch.equal(a2, torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda'))}")

# Test 3: Broken pattern with multiple load/store pairs
print("\n[TEST 3] Broken pattern (multiple load/store pairs)")
print("-" * 80)
a3 = torch.arange(N, dtype=torch.float32, device='cuda')
print(f"Before: {a3}")
s112_pattern_broken[(1,)](a3, N, BLOCK_SIZE=256)
print(f"After:  {a3}")
print(f"Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (same as input)")
print(f"Correct? {torch.equal(a3, torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda'))}")

if not torch.equal(a3, a1):
    print(f"✗ DIFFERENT RESULTS! This pattern is BROKEN!")
else:
    print(f"✓ Same as safe pattern (Triton may have optimized it)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The s112 implementation works because:

1. **Single Load/Store Sequence**:
   The kernel does exactly ONE tl.load followed by ONE tl.store

2. **SIMT Implicit Barrier**:
   ALL threads complete tl.load BEFORE ANY thread begins tl.store
   This is guaranteed by Triton/CUDA's SIMT execution model

3. **Read-Before-Write**:
   Because loads complete first, each thread reads the OLD value
   before any thread writes the NEW value

4. **Why it's fragile**:
   - Adding multiple load/store pairs breaks the guarantee
   - Compiler optimizations could reorder operations
   - Different GPU architectures may behave differently

5. **Why explicit sequencing is better**:
   The s112_triton_correct.py approach with sequential chunk
   launches is more explicit, portable, and maintainable, even
   though this "accidental correctness" happens to work.

This is a case of "undefined behavior that happens to work" due to
implementation details of Triton's code generation and CUDA's SIMT model.
""")
