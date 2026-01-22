import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all values with infinity as default for out-of-bounds
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Find minimum across the block
    block_min = tl.min(vals, axis=0)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.atomic_min(result_ptr, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize result with first element
    result = torch.tensor([a[0].item()], dtype=a.dtype, device=a.device)
    
    # Launch kernel with enough blocks to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s316_kernel[grid](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()