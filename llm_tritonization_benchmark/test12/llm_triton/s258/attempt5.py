import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize s as a vector to maintain type consistency
    s = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    
    for i in range(n_elements):
        # Load current element
        current_offset = i
        mask = current_offset < n_elements
        
        a_val = tl.load(a_ptr + current_offset, mask=mask, other=0.0)
        d_val = tl.load(d_ptr + current_offset, mask=mask, other=0.0)
        c_val = tl.load(c_ptr + current_offset, mask=mask, other=0.0)
        aa_val = tl.load(aa_ptr + current_offset, mask=mask, other=0.0)
        
        # Update s if condition is met - broadcast scalar values to match s shape
        cond = a_val > 0.0
        s_update = d_val * d_val
        s = tl.where(cond, tl.full([BLOCK_SIZE], s_update, dtype=tl.float32), s)
        
        # Compute outputs using broadcast scalar values
        b_val = tl.full([BLOCK_SIZE], s_update if cond else s[0], dtype=tl.float32)[0] * c_val + d_val
        e_val = (tl.full([BLOCK_SIZE], s_update if cond else s[0], dtype=tl.float32)[0] + 1.0) * aa_val
        
        # Store results
        tl.store(b_ptr + current_offset, b_val, mask=mask)
        tl.store(e_ptr + current_offset, e_val, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s258_kernel[grid](
        a, aa, b, c, d, e, n_elements, BLOCK_SIZE
    )