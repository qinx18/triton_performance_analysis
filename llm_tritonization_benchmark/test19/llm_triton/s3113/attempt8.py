import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element's absolute value
    first_val = tl.load(a_ptr)
    block_max = tl.abs(first_val)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        current_max = tl.max(abs_vals)
        block_max = tl.maximum(block_max, current_max)
    
    tl.store(result_ptr, block_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s3113_kernel[(1,)](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    return result.item()