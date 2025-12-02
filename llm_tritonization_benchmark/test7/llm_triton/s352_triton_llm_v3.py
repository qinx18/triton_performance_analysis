import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot = 0.0
    
    for i in range(0, n_elements, 5):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Load 5 elements at a time (unrolled)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block
        products = a_vals * b_vals
        dot += tl.sum(products)
        
        # Handle the remaining 4 elements manually if needed
        if i + 5 < n_elements:
            for offset in range(1, min(5, n_elements - i)):
                if i + offset < n_elements:
                    a_val = tl.load(a_ptr + i + offset)
                    b_val = tl.load(b_ptr + i + offset)
                    dot += a_val * b_val
    
    tl.store(dot_ptr, dot)

def s352_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Create output tensor for dot product result
    dot = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    
    s352_kernel[grid](
        a, b, dot,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot.item()