import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_max = float('-inf')
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max_vals = tl.max(abs_vals, axis=0)
        
        block_max = tl.maximum(block_max, block_max_vals)
    
    tl.store(result_ptr, block_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    s3113_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()