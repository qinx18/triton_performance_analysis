# S118 Analysis - Sequential Kernel Launches + Relative Tolerance

## Summary

**s118_triton_llm.py has TWO issues:**
1. **LLM bug**: Launches kernels in parallel instead of sequentially (race condition)
2. **Test issue**: Uses absolute tolerance instead of relative tolerance

The corrected version (s118_triton_correct.py) fixes the race condition and uses relative tolerance.

## The Algorithm

```python
# Original C code
for (int i = 1; i < LEN_2D; i++) {
    for (int j = 0; j <= i - 1; j++) {
        a[i] += bb[j][i] * a[i-j-1];
    }
}
```

## Issue #1: Race Condition (LLM Bug)

### The Problem

**LLM version** launches all `i` values in parallel:
```python
# All kernels run simultaneously
grid = (triton.cdiv(num_programs, 1),)
s118_kernel[grid](...)
```

**Data dependency**: Kernel for `i=3` reads `a[2]`, while kernel for `i=2` writes to `a[2]`.

**Access pattern example (N=5)**:
| i | Reads from | Writes to | Conflict? |
|---|------------|-----------|-----------|
| 1 | a[0] | a[1] | - |
| 2 | a[0], a[1] | a[2] | ✗ Reads a[1] that i=1 writes! |
| 3 | a[0], a[1], a[2] | a[3] | ✗ Reads a[2] that i=2 writes! |
| 4 | a[0], a[1], a[2], a[3] | a[4] | ✗ Reads a[3] that i=3 writes! |

This is a **Read-After-Write (RAW) hazard** requiring sequential execution.

### The Fix

**Corrected version** launches kernels SEQUENTIALLY for each `i`:
```python
# Process each i value sequentially
for i in range(1, n):
    grid_size = triton.cdiv(i, BLOCK_SIZE)
    s118_kernel[(grid_size,)](a_ptr=a, bb_ptr=bb, n=n, i_val=i, BLOCK_SIZE=BLOCK_SIZE)
```

Each kernel parallelizes over `j` values, which is safe since they don't have dependencies within the same `i`.

## Issue #2: Relative Tolerance (Test Issue)

### The Problem

s118 performs accumulation with growing values:
```
a[i] += sum(bb[j, i] * a[i-j-1] for j in 0..i-1)
```

For N=100:
- Values grow to ~1.34e+10
- Absolute error: 71,680 (looks bad!)
- Relative error: 9.15e-05 = 0.00915% (excellent!)

**Test used**: `max_error < 1e-3` (absolute) ❌ FAILS

### The Fix

Use **relative tolerance** appropriate for accumulating operations:
```python
passed = torch.allclose(pytorch_result, triton_result, rtol=5e-4, atol=1e-6)
# rtol=5e-4: 0.05% relative error for N sequential accumulations
# atol=1e-6: For near-zero values
```

## Test Results

### Before Fix (s118_triton_llm.py)
```
Would have race conditions and unpredictable results
```

### After Fix (s118_triton_correct.py) with Absolute Tolerance
```
Testing N=    10... ✓ PASS  (max_err=4.77e-07)
Testing N=    50... ✗ FAIL  (max_error=1.56e-02)
Testing N=   100... ✗ FAIL  (max_error=3.15e+06)
```

### After Fix with Relative Tolerance
```
Testing N=    10... ✓ PASS  (max_rel_err=4.25e-07)
Testing N=    50... ✓ PASS  (max_rel_err=1.75e-05)
Testing N=   100... ✓ PASS  (max_rel_err=1.22e-05)
```

## Implementation Details

### Sequential Kernel Launch
Each `i` value is processed in order:
- `i=1`: Launch kernel to process j=[0]
- `i=2`: Launch kernel to process j=[0,1]
- `i=3`: Launch kernel to process j=[0,1,2]
- ...
- `i=N-1`: Launch kernel to process j=[0,1,...,N-2]

### Parallelization Within Each Kernel
For each `i`, the `j` loop is parallelized:
```python
# Parallel over j values
j_offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = j_offsets < i_val

# Load bb[j, i] and a[i-j-1]
bb_vals = tl.load(bb_ptr + j_offsets * n + i_val, mask=mask)
a_vals = tl.load(a_ptr + i_val - j_offsets - 1, mask=mask)

# Compute and sum
products = bb_vals * a_vals
result = tl.sum(products, axis=0)

# Atomic add to accumulate
tl.atomic_add(a_ptr + i_val, result)
```

## Performance Considerations

**Drawback**: Sequential kernel launches add overhead:
- N=10: Very fast
- N=50: Moderate
- N=100: 100 sequential launches
- N=1000: Would be slow (1000 launches)

**Why necessary**: The RAW dependencies cannot be avoided - each `a[i]` depends on all previous `a[0..i-1]` values.

**Alternative approach**: Could use a single kernel with intra-block synchronization, but that limits parallelism and is more complex.

## Verdict

**Issue #1**: Genuine LLM algorithm bug - incorrect parallelization strategy
**Issue #2**: Test infrastructure issue - wrong tolerance metric

**Category**:
- Algorithm error (race condition due to incorrect parallelization)
- Test issue (absolute vs relative tolerance)

Both fixes required for correctness.
