import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        partial_sum += tl.sum(a_vals)
    
    tl.store(result_ptr, partial_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    s311_kernel[(1,)](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()