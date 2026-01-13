import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_dot = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        block_dot += tl.sum(a_vals * b_vals, axis=0)
    
    tl.store(output_ptr, block_dot)

def s313_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor for the dot product result
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single thread block since we need global reduction
    grid = (1,)
    
    s313_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()