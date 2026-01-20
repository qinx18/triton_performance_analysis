import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, partial_results_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals)
    
    tl.store(partial_results_ptr + pid, block_max)

def s3113_triton(a):
    n = a.shape[0]
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    partial_results = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    
    s3113_kernel[grid](
        a, partial_results, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    final_max = torch.max(partial_results)
    return final_max