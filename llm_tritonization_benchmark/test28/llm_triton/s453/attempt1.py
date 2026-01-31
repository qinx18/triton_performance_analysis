import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation - each iteration depends on previous
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute s values: s = 2 * (i + 1) for each position
        s_vals = 2.0 * (current_offsets + 1)
        
        # Compute a[i] = s * b[i]
        a_vals = s_vals * b_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s453_kernel[(1,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a