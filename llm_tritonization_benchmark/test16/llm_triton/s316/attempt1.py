import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate how many blocks we need to process all elements
    n_blocks = tl.cdiv(n_elements - 1, BLOCK_SIZE)  # Start from index 1
    
    # Initialize with a[0]
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, first_val)
    
    # Process remaining elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    block_min = float('inf')
    
    for block_idx in range(n_blocks):
        block_start = 1 + block_idx * BLOCK_SIZE  # Start from index 1
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.minimum(block_min, tl.min(vals, axis=0))
    
    # Compare with a[0] and store final result
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        final_min = tl.minimum(first_val, block_min)
        tl.store(result_ptr, final_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    s316_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()