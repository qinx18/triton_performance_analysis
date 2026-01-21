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
    
    # Compute base values
    base_vals = b_vals + d_vals * e_vals
    cond_vals = c_vals + d_vals * e_vals
    
    # Create condition mask
    c_positive = c_vals > 0.0
    
    # For each i, we need to write to j and possibly j+1
    # j starts at -1, so j = i after first increment
    j_base = i_offsets  # j after first increment
    j_cond = i_offsets * 2 + 1  # j after second increment when condition is true
    
    # Store base values (always executed)
    base_mask = i_mask
    tl.store(a_ptr + j_base, base_vals, mask=base_mask)
    
    # Store conditional values (only when c[i] > 0)
    cond_mask = i_mask & c_positive
    tl.store(a_ptr + j_cond, cond_vals, mask=cond_mask)

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