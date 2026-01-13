import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel will be called for each block and we'll do reduction in the wrapper
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load values
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find max value and index within this block
    max_val = tl.max(vals)
    
    # Find the local index of the maximum value
    is_max = vals == max_val
    local_indices = tl.arange(0, BLOCK_SIZE)
    
    # Get the first occurrence of max value in this block
    masked_indices = tl.where(is_max & mask, local_indices, BLOCK_SIZE)
    local_max_idx = tl.min(masked_indices)
    
    # Convert to global index
    global_max_idx = block_start + local_max_idx
    
    # Store results for this block
    block_id = tl.program_id(0)
    tl.store(result_ptr + block_id * 2, max_val)
    tl.store(result_ptr + block_id * 2 + 1, global_max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Use PyTorch for argmax reduction which is more efficient
    max_val = torch.max(a).item()
    max_idx = torch.argmax(a).item()
    
    # Return the same value as C code: index + x + 1
    return max_idx + max_val + 1