import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Find minimum across the block
    min_val = tl.min(vals)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # For reduction, we need to handle it differently
    # Use a two-stage reduction approach
    BLOCK_SIZE = 1024
    
    if n_elements <= BLOCK_SIZE:
        # Single block can handle all elements
        s316_kernel[(1,)](a, result, n_elements, BLOCK_SIZE)
    else:
        # Multi-stage reduction
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        temp_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
        
        # First stage: reduce within each block
        s316_first_stage_kernel[(num_blocks,)](a, temp_results, n_elements, BLOCK_SIZE)
        
        # Second stage: reduce the temporary results
        if num_blocks <= BLOCK_SIZE:
            s316_kernel[(1,)](temp_results, result, num_blocks, BLOCK_SIZE)
        else:
            # For very large arrays, we might need more stages
            current_size = num_blocks
            current_data = temp_results
            
            while current_size > 1:
                next_blocks = triton.cdiv(current_size, BLOCK_SIZE)
                next_results = torch.zeros(next_blocks, dtype=a.dtype, device=a.device)
                
                s316_kernel[(next_blocks,)](current_data, next_results, current_size, BLOCK_SIZE)
                
                current_data = next_results
                current_size = next_blocks
            
            result = current_data
    
    return result.item()

@triton.jit
def s316_first_stage_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
    
    # Find minimum in this block
    min_val = tl.min(vals)
    
    # Store result for this block
    tl.store(result_ptr + block_id, min_val)