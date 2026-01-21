import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    block_max = tl.max(vals)
    tl.store(result_ptr + pid, block_max)

def s314_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n))
    
    # First reduction: find max in each block
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    grid = (num_blocks,)
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    s314_kernel[grid](a, block_results, n, BLOCK_SIZE)
    
    # Recursive reduction until single value
    while block_results.shape[0] > 1:
        n_remaining = block_results.shape[0]
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_remaining))
        num_blocks = triton.cdiv(n_remaining, BLOCK_SIZE)
        grid = (num_blocks,)
        next_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
        
        s314_kernel[grid](block_results, next_results, n_remaining, BLOCK_SIZE)
        block_results = next_results
    
    return block_results.item()