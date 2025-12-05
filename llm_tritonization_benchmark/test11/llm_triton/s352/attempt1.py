import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start position (each block processes BLOCK_SIZE*5 elements)
    elements_per_block = BLOCK_SIZE * 5
    block_start = pid * elements_per_block
    
    # Initialize dot accumulator
    dot_sum = 0.0
    
    # Process elements in chunks of 5
    for chunk_start in range(0, elements_per_block, 5):
        # Calculate actual indices for this chunk
        base_idx = block_start + chunk_start
        
        # Check if we have at least 5 elements remaining
        if base_idx + 4 < n_elements:
            # Load 5 consecutive elements from both arrays
            a_vals = tl.load(a_ptr + base_idx + tl.arange(0, 5))
            b_vals = tl.load(b_ptr + base_idx + tl.arange(0, 5))
            
            # Compute dot product for these 5 elements
            chunk_dot = tl.sum(a_vals * b_vals)
            dot_sum += chunk_dot
        else:
            break
    
    # Store the partial dot product
    tl.store(dot_ptr + pid, dot_sum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Ensure we process in multiples of 5
    effective_n = (n_elements // 5) * 5
    
    # Calculate number of blocks
    BLOCK_SIZE = 64
    elements_per_block = BLOCK_SIZE * 5
    num_blocks = triton.cdiv(effective_n, elements_per_block)
    
    # Create output tensor for partial sums
    partial_dots = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s352_kernel[(num_blocks,)](
        a, b, partial_dots, effective_n, BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = partial_dots.sum()
    
    return dot