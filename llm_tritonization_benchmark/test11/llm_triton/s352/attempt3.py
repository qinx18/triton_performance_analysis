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
    
    # Define offset vectors once
    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in chunks of 5
    for chunk_idx in range(5):
        # Calculate indices for this position in each chunk of 5
        chunk_offsets = block_start + base_offsets * 5 + chunk_idx
        
        # Create mask for valid elements
        mask = chunk_offsets < n_elements
        
        # Load elements from both arrays
        a_vals = tl.load(a_ptr + chunk_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + chunk_offsets, mask=mask, other=0.0)
        
        # Compute partial dot product
        chunk_dot = tl.sum(a_vals * b_vals)
        dot_sum += chunk_dot
    
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