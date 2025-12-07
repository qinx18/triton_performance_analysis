import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle first element: a[0] = (b[0] + b[LEN_1D-1]) * 0.5
    if tl.program_id(0) == 0:
        b_0 = tl.load(b_ptr)
        b_last = tl.load(b_ptr + n_elements - 1)
        result_0 = (b_0 + b_last) * 0.5
        tl.store(a_ptr, result_0)
    
    # Handle remaining elements in parallel: a[i] = (b[i] + b[i-1]) * 0.5
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] and b[i-1]
        b_i = tl.load(b_ptr + current_offsets, mask=mask)
        b_im1 = tl.load(b_ptr + current_offsets - 1, mask=mask)
        
        # Compute result
        result = (b_i + b_im1) * 0.5
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a