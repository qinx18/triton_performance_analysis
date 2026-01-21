import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load first element as initial value
    x = tl.load(a_ptr)
    
    # Process array in blocks to find maximum
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(vals)
        x = tl.maximum(x, block_max)
    
    return x

def s314_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single thread since we need global maximum
    grid = (1,)
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s314_kernel[grid](a, n, BLOCK_SIZE)
    
    # Calculate maximum using Triton reduction
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n))
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    @triton.jit
    def max_reduction_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(vals)
        tl.store(result_ptr + pid, block_max)
    
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    max_reduction_kernel[grid](a, block_results, n, BLOCK_SIZE)
    
    # Final reduction on CPU for small number of blocks
    return block_results.max().item()