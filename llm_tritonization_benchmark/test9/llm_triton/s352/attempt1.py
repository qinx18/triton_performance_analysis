import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize dot product accumulator
    dot = 0.0
    
    # Process in steps of 5 (unrolled dot product)
    for step in range(0, n_elements, 5 * BLOCK_SIZE):
        current_start = block_start + step
        
        # Process 5 elements at a time (unrolled)
        for unroll in range(5):
            element_offsets = current_start + unroll + offsets * 5
            mask = element_offsets < n_elements
            
            # Load elements
            a_vals = tl.load(a_ptr + element_offsets, mask=mask, other=0.0)
            b_vals = tl.load(b_ptr + element_offsets, mask=mask, other=0.0)
            
            # Accumulate dot product
            dot += tl.sum(a_vals * b_vals)
    
    # Store result
    tl.store(output_ptr + pid, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE * 5)  # Divide by 5 due to unrolling
    
    # Create output tensor for partial results
    output = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s352_kernel[(grid_size,)](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    result = torch.sum(output)
    
    return result.item()