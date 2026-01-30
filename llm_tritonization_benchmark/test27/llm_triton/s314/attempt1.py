import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    block_max = tl.full((BLOCK_SIZE,), first_val, dtype=tl.float32)
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.maximum(block_max, vals)
    
    # Reduce within block
    final_max = tl.max(block_max, axis=0)
    tl.store(result_ptr, final_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    s314_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()