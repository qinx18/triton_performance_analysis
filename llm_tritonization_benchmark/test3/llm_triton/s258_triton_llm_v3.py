import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s = 0.0
    
    for i in range(n_elements):
        # Load a[i] and check condition
        a_val = tl.load(a_ptr + i)
        
        # Update s if condition is met
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s = d_val * d_val
        
        # Load other values needed for this iteration
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        aa_val = tl.load(aa_ptr + i)  # aa[0][i]
        
        # Compute results for this iteration
        b_val = s * c_val + d_val
        e_val = (s + 1.0) * aa_val
        
        # Store results if within bounds for this thread block
        if i >= block_start and i < block_start + BLOCK_SIZE and i < n_elements:
            local_offset = i - block_start
            if local_offset < BLOCK_SIZE and block_start + local_offset < n_elements:
                tl.store(b_ptr + i, b_val)
                tl.store(e_ptr + i, e_val)

def s258_triton(a, b, c, d, e, aa):
    n_elements = a.shape[0]
    
    # Use a single thread block to maintain sequential dependency on s
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s258_kernel[grid](
        a, b, c, d, e, aa,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )