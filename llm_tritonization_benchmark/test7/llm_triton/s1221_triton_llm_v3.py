import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(4, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Load b[i-4] and a[i]
        b_prev_offsets = current_offsets - 4
        b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute b[i] = b[i-4] + a[i]
        result = b_prev_vals + a_vals
        
        # Store result
        tl.store(b_ptr + current_offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential processing due to WAR dependency
    grid = (1,)
    
    s1221_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b