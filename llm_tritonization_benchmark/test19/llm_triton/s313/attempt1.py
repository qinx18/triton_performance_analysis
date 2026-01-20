import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        block_dot = tl.sum(a_vals * b_vals)
        partial_sum += block_dot
    
    tl.store(result_ptr, partial_sum)

def s313_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s313_kernel[grid](
        a, b, result,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()