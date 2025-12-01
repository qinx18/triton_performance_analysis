import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[j] where j = i + 1
        a_j_offsets = current_offsets + 1
        a_j_mask = a_j_offsets < (n_elements + 1)  # Since we access up to LEN_1D
        a_j = tl.load(a_ptr + a_j_offsets, mask=a_j_mask, other=0.0)
        
        # Load b[i]
        b_i = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = a[j] + b[i]
        result = a_j + b_i
        
        # Store result to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s121_kernel[grid](a, b, n_elements, BLOCK_SIZE)