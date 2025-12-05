import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Read from the copy at LEN_1D/2 (broadcast to all threads)
    mid_index = n_elements // 2
    a_mid_val = tl.load(a_copy_ptr + mid_index)
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    s1113_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE
    )