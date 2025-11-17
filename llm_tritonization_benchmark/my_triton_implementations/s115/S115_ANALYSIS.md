# S115 Correctness Analysis

## Summary

**s115_triton_llm.py is CORRECT!** The apparent test failures were due to using **absolute tolerance** instead of **relative tolerance**.

## The Issue

Initial testing showed s115 "failing" at larger sizes:
- N=10: PASS (max_err=6.33e-08)
- N=50: FAIL (max_error=1.25e+00)
- N=100: FAIL (max_error=3.36e+07)

However, these "failures" were due to improper error measurement.

## Root Cause: Exponential Value Growth

### What is s115?

s115 implements **back substitution**, a triangular solve algorithm:

```python
for j in range(N):
    for i in range(j + 1, N):
        a[i] -= aa[j, i] * a[j]
```

### Why Values Grow Exponentially

With certain input matrices (especially random matrices), back substitution can cause values to grow dramatically:

| N | Initial Range | Final Max Value | Value Growth |
|---|---------------|-----------------|--------------|
| 10 | ~[-2, 2] | ~7.8 | 4x |
| 50 | ~[-2, 2] | ~1.06e+08 | 50 million x |
| 100 | ~[-2, 2] | ~2.30e+10 | 10 billion x |
| 200 | ~[-2, 2] | ~1.43e+24 | 10^24 x |

This is **mathematically correct behavior**, not a bug!

## Absolute vs Relative Error

### Absolute Error (Wrong Approach)

At N=50:
- PyTorch result[46] = 1.0586280800e+08
- Triton result[46] = 1.0586279200e+08
- **Absolute error** = 16.0
- Test used: `max_error < 1e-3` ❌ FAILS

But an error of 16 on a value of 100 million is negligible!

### Relative Error (Correct Approach)

At N=50:
- Absolute error = 16.0
- **Relative error** = 16 / 1.0586e+08 = **1.43e-06** (0.00014%)
- Test should use: `torch.allclose(a, b, rtol=1e-4)` ✅ PASSES

## Test Results with Relative Tolerance

| N | Max Abs Error | Max Rel Error | Max Value | Relative Pass? |
|---|---------------|---------------|-----------|----------------|
| 10 | 1.19e-07 | 7.83e-08 | 7.83e+00 | ✅ PASS |
| 20 | 9.16e-05 | 4.03e-07 | 2.79e+02 | ✅ PASS |
| 30 | 1.22e-04 | 9.72e-07 | 1.02e+03 | ✅ PASS |
| 50 | **1.60e+01** | **1.43e-06** | 1.06e+08 | ✅ PASS |
| 100 | **6.14e+03** | **5.79e-06** | 2.30e+10 | ✅ PASS |
| 200 | **1.01e+18** | **7.82e-06** | 1.43e+24 | ✅ PASS |

All tests pass with relative tolerance `rtol=1e-4` (0.01%)!

## Implementation Correctness

The Triton implementation correctly:

1. **Sequential j-loop**: Processes columns sequentially (required for data dependencies)
2. **Parallel i-loop**: Parallelizes independent row updates within each column
3. **Correct indexing**: `a[i] -= aa[j, i] * a[j]` matches baseline
4. **Grid calculation**: Properly divides work across thread blocks

```python
# Sequential over j (columns)
for j in range(LEN_2D):
    num_elements = LEN_2D - (j + 1)
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)

    if grid_size > 0:
        # Parallel over i (rows where i > j)
        s115_kernel[(grid_size,)](a, aa, LEN_2D, j, BLOCK_SIZE)
```

## Why Relative Tolerance Matters

For algorithms with accumulating operations (like back substitution):

1. **Floating-point arithmetic** introduces small errors at each step
2. **Error accumulation** grows with the number of operations
3. **Value magnitude** affects absolute error but not relative error
4. **Relative error** is the appropriate metric for numerical correctness

## Fix Applied

Updated `test_s115_correctness.py`:

```python
# Old (incorrect)
max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
if max_error < 1e-3:  # Absolute tolerance
    print("PASS")

# New (correct)
passed = torch.allclose(pytorch_result, triton_result, rtol=1e-4, atol=1e-6)
# rtol=1e-4: 0.01% relative error allowed
# atol=1e-6: Small absolute tolerance for near-zero values
```

## Conclusion

**s115_triton_llm.py is a correct implementation!**

- Implementation matches baseline algorithm exactly
- Sequential/parallel decomposition is correct
- All tests pass with appropriate relative tolerance
- Exponential value growth is expected behavior, not a bug
- Test infrastructure needed fixing, not the implementation

## Lessons Learned

1. **Always use relative tolerance** for algorithms with:
   - Accumulating operations (loops, recurrences)
   - Value growth (exponentials, matrix operations)
   - Multiple sequential dependencies

2. **Absolute tolerance is appropriate** only when:
   - Values remain in a known bounded range
   - Single-pass algorithms with no accumulation
   - Measuring differences near zero

3. **For TSVC benchmarks**, many functions (triangular solves, reductions, etc.) need relative tolerance, not absolute.

## Recommendations

Other TSVC functions that likely need relative tolerance:
- s114, s118, s119 (triangular operations)
- s116, s117 (matrix-vector operations)
- Functions with iteration-dependent accumulation
- Any function where values can grow unbounded
