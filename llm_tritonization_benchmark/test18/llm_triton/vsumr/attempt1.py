import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum += tl.sum(vals)
    
    tl.store(output_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use a single thread block to compute the full reduction
    BLOCK_SIZE = 1024
    
    vsumr_kernel[(1,)](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()