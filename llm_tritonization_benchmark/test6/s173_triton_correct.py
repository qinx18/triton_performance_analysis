import torch
import triton
import triton.language as tl

@triton.jit
def s173_kernel(
    a_ptr, b_ptr,
    n_elements, k, start_i,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s173: a[i+k] = a[i] + b[i]
    Processes from start_i (for sequential launches when dependencies exist)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = start_i + block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices within half_len
    mask = offsets < n_elements

    # Load a[i] and b[i] with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute a[i] + b[i]
    result = a_vals + b_vals

    # Store to a[i+k] with bounds checking
    store_offsets = offsets + k
    store_mask = mask & (store_offsets < (n_elements * 2))  # Original array size
    tl.store(a_ptr + store_offsets, result, mask=store_mask)

def s173_triton(a, b, k):
    """
    Corrected Triton implementation of TSVC s173.

    When k >= half_len: No dependencies, can parallelize fully.
    When k < half_len: RAW dependencies exist, must use sequential launches.

    Example with k=5, half_len=10:
    - Read range: a[0:10]
    - Write range: a[5:15]
    - Overlap: a[5:10] (both read and written)
    - Must execute i=0,1,2,3,4 before i=5,6,7,8,9 to get correct values
    """
    a = a.contiguous()
    b = b.contiguous()

    len_1d = a.size(0)
    half_len = len_1d // 2

    BLOCK_SIZE = 256

    if k >= half_len:
        # No dependencies: read [0, half_len), write [k, k+half_len)
        # These ranges don't overlap, can run fully parallel
        grid = (triton.cdiv(half_len, BLOCK_SIZE),)
        s173_kernel[grid](
            a, b,
            half_len, k, 0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Dependencies exist: must execute sequentially
        # Process in chunks of size k to maintain dependencies
        for start_i in range(0, half_len, k):
            # Each chunk can be parallelized internally
            chunk_size = min(k, half_len - start_i)
            grid = (triton.cdiv(chunk_size, BLOCK_SIZE),)
            s173_kernel[grid](
                a, b,
                start_i + chunk_size, k, start_i,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return a
