import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot = 0.0
    
    for block_start in range(0, n_elements, 5):
        # Load 5 elements at a time
        current_offsets = block_start + tl.arange(0, 5)
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block of 5
        products = a_vals * b_vals
        dot += tl.sum(products)
    
    # Store the result
    tl.store(dot_ptr, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor for the dot product result
    dot_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s352_kernel[grid](
        a, b, dot_result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot_result[0].item()