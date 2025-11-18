# S119 Analysis - Diagonal Dependency Sequential Launches

## Summary

**s119_triton_llm.py has a race condition bug.** The LLM launches all elements in parallel, but the diagonal dependency pattern requires sequential processing. The corrected version (s119_triton_correct.py) passes all tests.

## The Algorithm

```python
# Original C code
for (int i = 1; i < LEN_2D; i++) {
    for (int j = 1; j < LEN_2D; j++) {
        aa[i][j] = aa[i-1][j-1] + bb[i][j];
    }
}
```

This is a **diagonal dependency pattern**: each element `aa[i,j]` depends on `aa[i-1,j-1]`.

## Issue: Race Condition (LLM Bug)

### The Problem

**LLM version** launches all elements in parallel:
```python
# Linearizes all (i,j) pairs and processes in one kernel launch
total_elements = (M - 1) * (N - 1)
grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
s119_kernel[(grid_size,)](...)  # All elements processed simultaneously
```

**Diagonal dependency visualization** (5x5 array):
```
     j=0  j=1  j=2  j=3  j=4
i=0   -    -    -    -    -
i=1   -   (1)  (2)  (3)  (4)
i=2   -   (2)  (3)  (4)  (5)
i=3   -   (3)  (4)  (5)  (6)
i=4   -   (4)  (5)  (6)  (7)
```
Numbers in parentheses show dependency depth (how many diagonal steps from origin).

**Race condition example**:
- Thread processing `aa[2,2]` reads `aa[1,1]`
- Thread processing `aa[1,1]` writes `aa[1,1]`
- These can execute in any order → **Read-After-Write (RAW) hazard**

### Dependency Pattern

Elements along each diagonal can be computed in parallel, but diagonals must be computed sequentially:

| Wave | Elements that can be parallel |
|------|-------------------------------|
| 0 | aa[1,1] |
| 1 | aa[1,2], aa[2,1] |
| 2 | aa[1,3], aa[2,2], aa[3,1] |
| 3 | aa[1,4], aa[2,3], aa[3,2], aa[4,1] |
| ... | ... |

**Alternative approach**: Process row-by-row sequentially (simpler than diagonal waves).

### The Fix

**Corrected version** launches kernels SEQUENTIALLY for each row `i`:
```python
# Process each row i sequentially
for i in range(1, M):
    grid_size = triton.cdiv(N - 1, BLOCK_SIZE)
    s119_kernel[(grid_size,)](aa_ptr=aa, bb_ptr=bb, ..., i_val=i, ...)
```

Each kernel parallelizes over the `j` dimension (columns), which is safe since:
- For fixed `i`, elements `aa[i,1]`, `aa[i,2]`, ..., `aa[i,N-1]` depend on `aa[i-1,0]`, `aa[i-1,1]`, ..., `aa[i-1,N-2]`
- Row `i-1` was already fully computed in the previous kernel launch
- No RAW hazards within a single row

## Test Results

### Before Fix (s119_triton_llm.py)
```
Would have unpredictable results due to race conditions
```

### After Fix (s119_triton_correct.py)
```
Testing N=    10... ✓ PASS  (max_rel_err=0.00e+00)
Testing N=    50... ✓ PASS  (max_rel_err=0.00e+00)
Testing N=   100... ✓ PASS  (max_rel_err=0.00e+00)
```

Perfect match! Unlike s115 and s118, s119 doesn't have exponential value growth, so the results match exactly.

## Implementation Details

### Sequential Row Processing
Each row `i` is processed in order:
- `i=1`: Launch kernel to compute `aa[1,1..N-1]`
- `i=2`: Launch kernel to compute `aa[2,1..N-1]`
- `i=3`: Launch kernel to compute `aa[3,1..N-1]`
- ...
- `i=M-1`: Launch kernel to compute `aa[M-1,1..N-1]`

### Parallelization Within Each Row
For each row `i`, the `j` loop is parallelized:
```python
# Parallel over j values (columns)
j_offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
mask = j_offsets < N

# Load aa[i-1, j-1] (from previous row, already computed)
aa_prev_addrs = aa_ptr + (i_val - 1) * stride_aa_0 + (j_offsets - 1) * stride_aa_1
aa_prev = tl.load(aa_prev_addrs, mask=mask)

# Load bb[i, j]
bb_curr = tl.load(bb_curr_addrs, mask=mask)

# Compute and store aa[i, j] = aa[i-1, j-1] + bb[i, j]
result = aa_prev + bb_curr
tl.store(aa_curr_addrs, result, mask=mask)
```

## Why Relative Tolerance?

Unlike s115 and s118, s119 doesn't have exponential value growth because:
- It's just addition: `aa[i,j] = aa[i-1,j-1] + bb[i,j]`
- No multiplication or accumulation that compounds errors
- Values grow linearly at most

**Result**: Perfect numerical match (0 relative error) for all test sizes!

## Performance Considerations

**Sequential launches**: For NxN matrix, requires N-1 sequential kernel launches
- N=10: 9 launches (very fast)
- N=50: 49 launches (moderate)
- N=100: 99 launches (acceptable)
- N=1000: 999 launches (slower but necessary)

**Why necessary**: The diagonal dependency pattern cannot be avoided. Alternative diagonal-wave processing would be more complex with similar performance.

## Comparison with s118

| Aspect | s118 | s119 |
|--------|------|------|
| Dependency | `a[i]` depends on `a[i-j-1]` | `aa[i,j]` depends on `aa[i-1,j-1]` |
| Pattern | 1D accumulation | 2D diagonal |
| Sequential over | i (rows) | i (rows) |
| Parallel over | j (within each i) | j (columns, within each i) |
| Value growth | Exponential (needs rtol) | Linear (exact match) |
| Sequential launches | N | N |

Both require the same sequential processing strategy!

## Verdict

**LLM Algorithm Bug**: Incorrect parallelization - launches all elements in parallel despite diagonal RAW dependencies.

**Fix**: Sequential row-by-row processing with parallelization over columns.

**Category**: Algorithm error - race condition due to incorrect parallelization of dependent operations.

**Test Result**: Perfect correctness after fix (0 relative error).
