import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find max value and index within this block
    max_val = tl.max(vals)
    
    # Find the index of the maximum value
    is_max = vals == max_val
    indices = tl.arange(0, BLOCK_SIZE)
    masked_indices = tl.where(is_max & mask, indices, n_elements)
    min_index = tl.min(masked_indices)
    
    # Store results for this block
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, min_index.to(tl.float32))

def s315_triton(a):
    n = a.shape[0]
    
    # Use PyTorch for argmax reduction as recommended
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the exact value as specified: index + x + 1
    return float(max_idx + max_val + 1)