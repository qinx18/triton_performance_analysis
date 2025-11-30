import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check if any elements are negative
    is_negative = a_vals < 0.0
    
    # Get indices where condition is true
    indices = tl.where(is_negative, offsets, -1)
    
    # Store indices for reduction
    tl.store(result_ptr + offsets, indices, mask=mask)

@triton.jit
def s331_reduce_kernel(indices_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    indices = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    
    # Find the maximum valid index in this block
    max_idx = tl.max(indices)
    
    # Store block result
    if tl.program_id(0) == 0 and offsets[0] == 0:
        tl.store(result_ptr, max_idx)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary storage for indices
    temp_indices = torch.full((n_elements,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel to find negative indices
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s331_kernel[grid](a, temp_indices, n_elements, BLOCK_SIZE)
    
    # Find the last (maximum) valid index
    valid_indices = temp_indices[temp_indices >= 0]
    if valid_indices.numel() > 0:
        j = valid_indices.max().item()
    else:
        j = -1
    
    return j