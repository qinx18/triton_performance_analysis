import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
    
    tl.store(sum_ptr, sum_val)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    vsumr_kernel[grid](
        a, sum_result, n_elements, BLOCK_SIZE
    )
    
    return sum_result.item()