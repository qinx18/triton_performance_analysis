import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[j] where j = i + 1, so we need a[i+1]
        a_j_offsets = current_offsets + 1
        a_j_mask = a_j_offsets < (n_elements + 1)  # a has one more element than we're processing
        a_j_vals = tl.load(a_ptr + a_j_offsets, mask=a_j_mask)
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[j] + b[i] = a[i+1] + b[i]
        result = a_j_vals + b_vals
        
        # Store result to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = len(a) - 1  # Process LEN_1D-1 elements
    BLOCK_SIZE = 256
    
    # Launch kernel
    s121_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a