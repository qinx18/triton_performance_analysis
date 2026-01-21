import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Check bounds for input arrays
    i_mask = i_offsets < n_half
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=i_mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=i_mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Compute values
    first_vals = b_vals + d_vals * e_vals
    second_vals = c_vals + d_vals * e_vals
    
    # For each i, j starts at 2*i and potentially increments to 2*i+1
    j_first = i_offsets * 2
    j_second = i_offsets * 2 + 1
    
    # Create condition mask
    c_positive = c_vals > 0.0
    
    # Store first values (always executed when i < n_half)
    tl.store(a_ptr + j_first, first_vals, mask=i_mask)
    
    # Store second values (only when c[i] > 0 and i < n_half)
    second_mask = i_mask & c_positive
    tl.store(a_ptr + j_second, second_vals, mask=second_mask)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    # Use a block size that works well for this type of computation
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    # Launch kernel
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )