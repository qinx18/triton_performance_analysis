import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value and its index using reduction
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from array a
    values = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Create indices for this block
    indices = offsets
    
    # Find local maximum and its index
    local_max = tl.max(values, axis=0)
    local_max_mask = values == local_max
    
    # Get the first index where maximum occurs
    local_indices = tl.where(local_max_mask, indices, n_elements)
    local_min_idx = tl.min(local_indices, axis=0)
    
    # Store results (each block stores its local max and index)
    result_offset = pid * 2
    tl.store(result_ptr + result_offset, local_max)
    tl.store(result_ptr + result_offset + 1, local_min_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create result tensor to store max values and indices from each block
    result = torch.empty((num_blocks * 2,), dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s315_kernel[grid](
        a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce results from all blocks on CPU to find global maximum
    result_cpu = result.cpu().numpy()
    max_values = result_cpu[::2]  # Even indices contain max values
    max_indices = result_cpu[1::2].astype(int)  # Odd indices contain indices
    
    # Find global maximum
    global_max_idx = max_values.argmax()
    x = max_values[global_max_idx]
    index = max_indices[global_max_idx]
    
    # Handle case where multiple blocks have the same max value
    # Choose the one with smallest index
    same_max_mask = max_values == x
    if same_max_mask.sum() > 1:
        valid_indices = max_indices[same_max_mask]
        index = valid_indices.min()
    
    chksum = x + float(index)
    
    return torch.tensor([x, float(index), chksum], dtype=torch.float32, device=a.device)