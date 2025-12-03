import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        dot += tl.sum(products)
    
    # Store the result (only the first thread writes the final dot product)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, dot)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor for the dot product result
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    vdotr_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()