import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Update s for this block
        valid_count = tl.sum(mask.to(tl.int32))
        s += valid_count * 2.0
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a values
        a_vals = s * b_vals
        
        # Store a values
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since we need sequential s computation
    grid = (1,)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )