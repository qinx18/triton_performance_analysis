import torch
import triton
import triton.language as tl

@triton.jit
def s173_kernel(
    a_ptr, b_ptr,
    n_elements, k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s173: a[i+k] = a[i] + b[i] for i in [0, LEN_1D/2)
    Uses coalesced memory access and masking for edge cases
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices within half_len
    mask = offsets < n_elements
    
    # Load a[i] and b[i] with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k] with bounds checking
    store_offsets = offsets + k
    store_mask = mask & (store_offsets < (n_elements * 2))  # Original array size
    tl.store(a_ptr + store_offsets, result, mask=store_mask)

def s173_triton(a, b, k):
    """
    Triton implementation of TSVC s173
    Optimized with coalesced memory access and efficient block processing
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    half_len = len_1d // 2
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(half_len, BLOCK_SIZE),)
    
    # Launch kernel
    s173_kernel[grid](
        a, b,
        half_len, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a