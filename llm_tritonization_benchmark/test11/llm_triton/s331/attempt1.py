import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize j to -1
    j = -1
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are negative
        negative_mask = a_vals < 0.0
        
        # Update j for each negative element in this block
        for i in range(BLOCK_SIZE):
            if (block_start + i) < n_elements:
                if negative_mask & (1 << i):
                    j = block_start + i
    
    # Store the result
    tl.store(result_ptr, j)

@triton.jit
def s331_kernel_optimized(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process array in blocks to find last negative element
    last_negative_idx = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Find negative elements
        negative_mask = a_vals < 0.0
        
        # Update last_negative_idx for any negative elements in this block
        # We need to check from highest to lowest index in the block
        for local_idx in range(BLOCK_SIZE - 1, -1, -1):
            global_idx = block_start + local_idx
            if global_idx < n_elements:
                # Extract the bit for this position
                is_negative = (negative_mask >> local_idx) & 1
                if is_negative:
                    last_negative_idx = global_idx
                    break
    
    tl.store(result_ptr, last_negative_idx)

@triton.jit  
def s331_kernel_simple(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes entire array sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
        
    j = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        negative_mask = a_vals < 0.0
        
        # Check each element in order to find last negative
        for i in range(BLOCK_SIZE):
            global_idx = block_start + i
            if global_idx < n_elements:
                element_negative = (negative_mask >> i) & 1
                if element_negative:
                    j = global_idx
    
    tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Result tensor to store j
    result = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block since we need sequential processing
    grid = (1,)
    
    s331_kernel_simple[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()