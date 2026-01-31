import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_half,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load data for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # For serial execution within block to maintain j dependencies
    for idx in range(BLOCK_SIZE):
        if block_start + idx >= n_half:
            break
            
        i_pos = block_start + idx
        j_base = i_pos * 2  # Maximum possible j for this i
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        first_val = tl.load(b_ptr + i_pos) + tl.load(d_ptr + i_pos) * tl.load(e_ptr + i_pos)
        tl.store(a_ptr + j_base, first_val)
        
        # Conditional second assignment
        c_val = tl.load(c_ptr + i_pos)
        if c_val > 0.0:
            second_val = c_val + tl.load(d_ptr + i_pos) * tl.load(e_ptr + i_pos)
            tl.store(a_ptr + j_base + 1, second_val)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    # Process sequentially to maintain j dependencies
    BLOCK_SIZE = 1
    grid = (n_half,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )