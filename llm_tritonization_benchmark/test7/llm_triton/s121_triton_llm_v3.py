import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[j] where j = i + 1
        j_offsets = current_offsets + 1
        j_mask = j_offsets < (n_elements + 1)  # j can go up to n_elements
        a_j = tl.load(a_ptr + j_offsets, mask=j_mask)
        
        # Load b[i]
        b_i = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[j] + b[i]
        result = a_j + b_i
        
        # Store back to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )