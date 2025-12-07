import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value and its index using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, first_val)  # max value
        tl.store(result_ptr + 1, 0.0)    # max index
    
    # Process blocks
    block_start = tl.program_id(0) * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load block values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum and its index
    local_max = tl.max(vals)
    local_max_mask = vals == local_max
    local_indices = tl.where(local_max_mask, current_offsets, n_elements)
    local_max_idx = tl.min(local_indices)
    
    # Atomic update of global maximum
    tl.atomic_max(result_ptr, local_max)
    
    # Load current global max to check if we need to update index
    current_global_max = tl.load(result_ptr)
    if local_max >= current_global_max:
        tl.atomic_min(result_ptr + 1, local_max_idx)

@triton.jit  
def s315_sequential_kernel(a_ptr, result_ptr, n_elements):
    # Sequential implementation for exact C behavior
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Since we can't use loops, we'll use a single-threaded approach
    # Load all values and find maximum sequentially
    for i in range(n_elements):
        val = tl.load(a_ptr + i)
        if val > max_val:
            max_val = val
            max_idx = i
    
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

@triton.jit
def s315_reduction_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    ptrs = a_ptr + block_start + offsets
    mask = (block_start + offsets) < n_elements
    
    # Load values
    vals = tl.load(ptrs, mask=mask, other=float('-inf'))
    
    # Find block maximum
    block_max = tl.max(vals)
    
    # Find the first index where maximum occurs
    max_mask = vals == block_max
    indices = block_start + offsets
    masked_indices = tl.where(max_mask, indices, n_elements)
    block_max_idx = tl.min(masked_indices)
    
    # Store block results
    result_offset = pid * 2
    tl.store(result_ptr + result_offset, block_max)
    tl.store(result_ptr + result_offset + 1, block_max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    device = a.device
    
    # Use sequential approach for exact behavior match
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    if num_blocks == 1:
        # Single block - use reduction within block
        result = torch.zeros(2, dtype=torch.float32, device=device)
        
        grid = (1,)
        s315_reduction_kernel[grid](
            a, result, n_elements, BLOCK_SIZE
        )
        
        max_val = result[0]
        max_idx = int(result[1])
    else:
        # Multiple blocks - two-phase reduction
        block_results = torch.zeros(num_blocks * 2, dtype=torch.float32, device=device)
        
        grid = (num_blocks,)
        s315_reduction_kernel[grid](
            a, block_results, n_elements, BLOCK_SIZE
        )
        
        # CPU reduction of block results for exact behavior
        block_results_cpu = block_results.cpu()
        max_val = block_results_cpu[0]
        max_idx = int(block_results_cpu[1])
        
        for i in range(1, num_blocks):
            val = block_results_cpu[i * 2]
            idx = int(block_results_cpu[i * 2 + 1])
            if val > max_val:
                max_val = val
                max_idx = idx
    
    # Calculate chksum as in original code
    chksum = max_val + float(max_idx)
    
    return max_val, max_idx, chksum