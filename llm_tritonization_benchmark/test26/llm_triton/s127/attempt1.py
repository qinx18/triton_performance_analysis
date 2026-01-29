import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < n
    
    # Load data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute j indices (j starts at -1, then increments)
    # For i-th iteration: j = 2*i and j = 2*i + 1
    j0_offsets = 2 * i_offsets
    j1_offsets = 2 * i_offsets + 1
    
    # Compute values
    val0 = b_vals + c_vals * d_vals
    val1 = b_vals + d_vals * e_vals
    
    # Store results
    # Mask for j0 (even indices)
    j0_mask = mask & (j0_offsets < 2 * n)
    tl.store(a_ptr + j0_offsets, val0, mask=j0_mask)
    
    # Mask for j1 (odd indices)
    j1_mask = mask & (j1_offsets < 2 * n)
    tl.store(a_ptr + j1_offsets, val1, mask=j1_mask)

def s127_triton(a, b, c, d, e):
    # Get array dimensions
    N = b.shape[0] // 2  # Loop runs for LEN_1D/2 iterations
    
    # Block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s127_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a