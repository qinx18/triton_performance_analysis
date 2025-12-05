import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block sequentially to respect break condition
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load single elements
        mask_single = idx < n_elements
        b_val = tl.load(b_ptr + idx, mask=mask_single)
        c_val = tl.load(c_ptr + idx, mask=mask_single)
        a_val = tl.load(a_ptr + idx, mask=mask_single)
        
        # Check break condition first
        if c_val > b_val:
            break
            
        # Perform computation
        result = a_val + b_val * c_val
        
        # Store result
        tl.store(a_ptr + idx, result, mask=mask_single)

def s482_triton(a, b, c):
    n_elements = a.numel()
    
    # Use smaller block size to better handle early break conditions
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a