import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    len_half,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s127 operation.
    Each thread processes one iteration of the original loop.
    """
    # Get program ID and compute offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < len_half
    
    # Load input data with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute results for both output positions
    # j = 2 * i, so we write to positions 2*i and 2*i+1
    result1 = b_vals + c_vals * d_vals  # a[2*i]
    result2 = b_vals + d_vals * e_vals  # a[2*i+1]
    
    # Compute output offsets
    out_offsets1 = 2 * offsets      # Even positions
    out_offsets2 = 2 * offsets + 1  # Odd positions
    
    # Store results with masking
    tl.store(a_ptr + out_offsets1, result1, mask=mask)
    tl.store(a_ptr + out_offsets2, result2, mask=mask)

def s127_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s127.
    Optimized using parallelization across the half-length iterations.
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    len_half = len(b) // 2
    
    # Handle empty case
    if len_half == 0:
        return a
    
    # Choose block size and grid size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_half, BLOCK_SIZE),)
    
    # Launch kernel
    s127_kernel[grid](
        a, b, c, d, e,
        len_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a