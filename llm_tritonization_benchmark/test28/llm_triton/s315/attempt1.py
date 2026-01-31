import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value and its index across all blocks
    # Each block processes BLOCK_SIZE elements and returns local max/argmax
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    values = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum and its position within the block
    local_max = tl.max(values)
    
    # Find the index of the maximum value within this block
    is_max = values == local_max
    local_indices = tl.where(is_max, current_offsets, n_elements)
    local_argmax = tl.min(local_indices)  # Get the first occurrence
    
    # Store results: [max_value, argmax_index] for this block
    result_offset = block_id * 2
    tl.store(result_ptr + result_offset, local_max)
    tl.store(result_ptr + result_offset + 1, local_argmax.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Result tensor: each block stores [max_value, argmax_index]
    block_results = torch.empty(num_blocks * 2, dtype=torch.float32, device=a.device)
    
    # Launch kernel to find local max/argmax per block
    grid = (num_blocks,)
    s315_kernel[grid](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Extract max values and indices from block results
    max_values = block_results[::2]  # Even indices contain max values
    argmax_indices = block_results[1::2].int()  # Odd indices contain argmax positions
    
    # Find global maximum among block results
    global_max_block_idx = torch.argmax(max_values)
    x = max_values[global_max_block_idx]
    index = argmax_indices[global_max_block_idx]
    
    return index + x + 1