import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(
    a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize s for this block
    s = 0.0
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load single elements
        a_val = tl.load(a_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        aa_val = tl.load(aa_ptr + idx)
        
        # Update s if condition is met
        if a_val > 0.0:
            s = d_val * d_val
        
        # Compute outputs
        b_val = s * c_val + d_val
        e_val = (s + 1.0) * aa_val
        
        # Store results
        tl.store(b_ptr + idx, b_val)
        tl.store(e_ptr + idx, e_val)

def s258_triton(a, aa, b, c, d, e):
    n_elements = a.numel()
    
    # Use smaller block size to maintain dependency ordering
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s258_kernel[grid](
        a, aa, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )