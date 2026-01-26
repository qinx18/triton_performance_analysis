import triton
import triton.language as tl
import torch

@triton.jit
def s171_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Calculate a indices: i * inc
        a_indices = current_offsets * inc
        
        # Load a[i * inc]
        a_vals = tl.load(a_ptr + a_indices, mask=mask)
        
        # Compute a[i * inc] += b[i]
        result = a_vals + b_vals
        
        # Store back to a[i * inc]
        tl.store(a_ptr + a_indices, result, mask=mask)

def s171_triton(a, b, inc):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    s171_kernel[(1,)](a, b, inc, n_elements, BLOCK_SIZE)