import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value and its index using reduction
    # Each block processes BLOCK_SIZE elements and finds local max
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load elements for this block
    mask = (block_start + offsets) < n_elements
    indices = block_start + offsets
    values = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum and its index
    local_max = tl.max(values, axis=0)
    
    # Find the index of the maximum value
    max_mask = values == local_max
    # Use the first occurrence of max value
    local_indices = tl.where(max_mask, indices, n_elements)
    local_index = tl.min(local_indices, axis=0)
    
    # Store results: [max_value, index] pairs for each block
    result_offset = block_id * 2
    tl.store(result_ptr + result_offset, local_max)
    tl.store(result_ptr + result_offset + 1, local_index.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create result tensor to store [max_value, index] pairs from each block
    block_results = torch.empty((num_blocks * 2,), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s315_kernel[(num_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Reduce results from all blocks on CPU/GPU
    block_maxes = block_results[::2]  # Even indices are max values
    block_indices = block_results[1::2]  # Odd indices are index values
    
    # Find global maximum
    global_max_block_idx = torch.argmax(block_maxes)
    x = block_maxes[global_max_block_idx]
    index = int(block_indices[global_max_block_idx])
    
    chksum = x + float(index)
    
    return x, index, chksum