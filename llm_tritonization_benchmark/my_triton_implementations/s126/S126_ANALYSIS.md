# S126 Analysis - Fixed Grid Size Inefficiency

## Summary

**s126_triton_llm.py has a performance bug that causes timeouts:**
- Launches LEN_2D separate kernel programs (one per column)
- For N=10010: launches 10,010 programs with no vectorization
- Causes timeout (>120s) despite correct algorithm

The corrected version (s126_triton_correct.py) uses the s1119 pattern and completes in 1.01s.

## The Algorithm

```c
// Original C code
k = 1 + nn * nn;
for (int i = 1; i < nn; i++) {
    for (int j = 1; j < nn; j++) {
        k++;
        bb[j][i] = bb[j-1][i] + flat_2d_array[k] * cc[j][i];
    }
}
```

**Pattern**: Vertical dependency - bb[j,i] depends on bb[j-1,i]
- Each column can be processed independently (parallelizable across i)
- Rows must be processed sequentially (j loop has dependency)

## Bug: Inefficient Grid Size

### LLM Code (s126_triton_llm.py:54)
```python
BLOCK_SIZE = 64  # Not used effectively
grid = (LEN_2D,)  # Launch LEN_2D programs!
```

**Problem**: Launches one program per column

**Impact at N=10010**:
- Grid size: 10,010 programs
- Each program: handles 1 column (scalar operations)
- No SIMD vectorization within kernel
- Result: Timeout >120s

### Comparison with s1119 (Correct Pattern)

**s1119_triton_llm.py (FAST - 1.01s @ N=10010):**
```python
BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)  # For N=10010: 40 programs
```

**Grid size comparison for N=10010**:
- s126 LLM: **10,010 programs** (each handles 1 column)
- s1119: **40 programs** (each handles 256 columns in parallel)
- Ratio: s126 launches **251x more programs**

## Fix: Use s1119 Grid Pattern

### Corrected Version (s126_triton_correct.py)

**Key changes:**

1. **Grid size**: Launch fewer programs, each handling multiple columns
```python
BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
```

2. **Vectorized column processing**: Each program handles BLOCK_SIZE columns
```python
@triton.jit
def s126_kernel(...):
    col_block_id = tl.program_id(0)
    col_start = col_block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < LEN_2D

    for j in range(1, LEN_2D):  # Sequential row processing
        # Vectorized loads/stores for multiple columns
        k_indices = 1 + col_offsets * LEN_2D + (j - 1)
        bb_prev = tl.load(bb_ptr + (j-1)*LEN_2D + col_offsets, mask=col_mask)
        cc_val = tl.load(cc_ptr + j*LEN_2D + col_offsets, mask=col_mask)
        flat_val = tl.load(flat_2d_array_ptr + k_indices - 1, mask=col_mask)

        new_val = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + j*LEN_2D + col_offsets, new_val, mask=col_mask)
```

## Test Results

### Correctness
```
Testing N=    50... ✓ PASS  (max_err=1.91e-06)
Testing N=   100... ✓ PASS  (max_err=5.72e-06)
Testing N=   200... ✓ PASS  (max_err=7.63e-06)
```

Perfect accuracy!

### Performance Comparison at N=10010

| Implementation | Time | Grid Size | Vectorization |
|----------------|------|-----------|---------------|
| s126_triton_llm | >120s (timeout) | 10,010 programs | None |
| s126_triton_correct | 1.01s | 40 programs | 256 columns/program |
| s1119_triton_llm | 1.01s | 40 programs | 256 columns/program |
| s126_pytorch | >180s (timeout) | N/A | Sequential |

**Speedup**: >119x faster (corrected vs LLM version)

### Why Baseline Times Out

The PyTorch baseline s126_pytorch also times out at N=10010 (>180s) because:
1. Purely sequential execution (no parallelization)
2. N=10010 means 100+ million element array
3. Nested loops with O(N²) operations

The Triton corrected version is **>178x faster** than the baseline!

## Why This Pattern Works

1. **Efficient grid size**: Launch O(N/256) programs instead of O(N)
2. **SIMD vectorization**: Process 256 columns in parallel per program
3. **Vertical dependency**: Allows column parallelization (same as s1119)
4. **Sequential row processing**: In-kernel loop maintains j-dependency

## Lessons Learned

### When to Use s1119 Pattern (In-Kernel Loop)

**✅ Use when:**
- Vertical dependency (aa[i,j] depends on aa[i-1,j])
- Horizontal dependency with reduction (s118 with atomics)
- Pattern: Each column/element can be processed independently

**❌ Don't use when:**
- Diagonal dependency (aa[i,j] depends on aa[i-1,j-1])
- Cross-thread dependencies without atomics

### Grid Size Best Practices

**Bad:**
```python
grid = (LEN_2D,)  # One program per column - no vectorization
```

**Good:**
```python
BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)  # Vectorized column blocks
```

## Verdict

**LLM Bug**: Performance bug - inefficient grid size

**Root Cause**:
- Launched one program per column (10,010 programs for N=10010)
- No SIMD vectorization within kernel
- 251x excessive kernel launches vs optimal

**Fix**: Apply s1119 pattern - vectorized column blocks

**Category**: LLM performance bug - correct algorithm but inefficient parallelization strategy

**Status**: ✅ Fixed and passing all tests with >119x speedup
