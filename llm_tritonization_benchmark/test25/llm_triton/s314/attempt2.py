import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n
    
    # Load values with -inf as default for masked elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
    
    # Find maximum in this block
    block_max = tl.max(vals, axis=0)
    
    # Store the block maximum
    tl.store(result_ptr + pid, block_max)

def s314_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for block results
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s314_kernel[(num_blocks,)](
        a, block_results, n, BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    x = torch.max(block_results).item()
    
    return x