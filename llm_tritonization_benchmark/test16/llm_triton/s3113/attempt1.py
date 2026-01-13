import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    running_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        running_max = tl.maximum(running_max, abs_vals)
    
    block_max = tl.max(running_max, axis=0)
    tl.store(result_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s3113_kernel[(1,)](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()