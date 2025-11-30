import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value and its index using reduction
    pid = tl.program_id(0)
    
    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values with mask
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum and its index
    local_max = tl.max(vals)
    local_max_mask = vals == local_max
    
    # Get the first index where maximum occurs
    indices = tl.arange(0, BLOCK_SIZE)
    masked_indices = tl.where(local_max_mask & mask, offsets, n_elements)
    local_max_idx = tl.min(masked_indices)
    
    # Store results: [max_value, max_index] for each block
    result_offset = pid * 2
    tl.store(result_ptr + result_offset, local_max)
    tl.store(result_ptr + result_offset + 1, local_max_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create result tensor to store [max_value, max_index] pairs
    block_results = torch.empty((grid_size * 2,), device=a.device, dtype=a.dtype)
    
    # Launch kernel for block-wise reduction
    s315_kernel[(grid_size,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU for simplicity (small number of blocks)
    block_results_cpu = block_results.cpu()
    
    # Extract max values and indices from each block
    max_vals = block_results_cpu[::2]
    max_indices = block_results_cpu[1::2].int()
    
    # Find global maximum
    global_max_val = torch.max(max_vals).item()
    global_max_block = torch.argmax(max_vals).item()
    global_max_idx = max_indices[global_max_block].item()
    
    # Handle case where global_max_idx might be invalid due to masking
    if global_max_idx >= n_elements:
        # Find the actual index by checking the original array
        global_max_idx = torch.argmax(a).item()
    
    # Return the final result as in original C code
    return global_max_idx + global_max_val + 1