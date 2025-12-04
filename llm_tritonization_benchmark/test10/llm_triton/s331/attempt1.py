import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to process all elements and find the maximum index that satisfies the condition
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1 (no negative value found)
    max_idx = -1
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        neg_mask = vals < 0.0
        
        # For each negative element, update max_idx if current index is larger
        for i in range(BLOCK_SIZE):
            if mask[i] and neg_mask[i]:
                current_idx = block_start + i
                if current_idx > max_idx:
                    max_idx = current_idx
    
    # Store result (only thread 0 writes the result)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, max_idx)

@triton.jit 
def s331_kernel_optimized(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process one block per program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Check which elements are negative
    neg_mask = vals < 0.0
    valid_mask = mask & neg_mask
    
    # Find the highest index in this block that is negative
    max_idx = -1
    for i in range(BLOCK_SIZE):
        if valid_mask[i]:
            idx = block_start + i
            if idx > max_idx:
                max_idx = idx
    
    # Store the result for this block
    block_results = result_ptr + pid
    tl.store(block_results, max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Use optimized version with reduction
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary storage for block results
    block_results = torch.full((num_blocks,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    s331_kernel_optimized[(num_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Find maximum across all blocks
    j = torch.max(block_results).item()
    
    return j