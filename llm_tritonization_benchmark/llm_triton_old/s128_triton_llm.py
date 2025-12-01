import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    half_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s128 computation.
    Processes the loop with data dependencies by computing sequentially.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_len
    
    # Load indices for this block
    i_vals = offsets
    
    # Compute k values: k = 2*i (since j starts at -1 and increments by 2 each iteration)
    k_vals = 2 * i_vals
    
    # Load data with masking
    d_vals = tl.load(d_ptr + i_vals, mask=mask)
    b_k_vals = tl.load(b_ptr + k_vals, mask=mask)
    c_k_vals = tl.load(c_ptr + k_vals, mask=mask)
    
    # Compute: a[i] = b[k] - d[i]
    a_vals = b_k_vals - d_vals
    
    # Compute: b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_k_vals
    
    # Store results with masking
    tl.store(a_ptr + i_vals, a_vals, mask=mask)
    tl.store(b_ptr + k_vals, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    """
    Triton implementation of TSVC s128.
    Optimized GPU version with coalesced memory access patterns.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    len_1d = len(a)
    half_len = len_1d // 2
    
    if half_len == 0:
        return a, b
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(half_len, BLOCK_SIZE),)
    
    # Launch kernel
    s128_kernel[grid](
        a, b, c, d,
        half_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b