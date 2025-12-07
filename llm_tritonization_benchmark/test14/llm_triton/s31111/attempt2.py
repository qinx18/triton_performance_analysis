import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Define offsets once
    offsets = tl.arange(0, 4)
    
    # Process 8 groups: a[0:4], a[4:8], a[8:12], ..., a[28:32]
    for group in range(8):
        base_idx = group * 4
        element_offsets = base_idx + offsets
        mask = element_offsets < n_elements
        
        elements = tl.load(a_ptr + element_offsets, mask=mask, other=0.0)
        group_sum = tl.sum(elements)
        sum_val += group_sum
    
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Single output value
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    
    # Single thread processes all groups
    grid = (1,)
    s31111_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()