import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block boundaries
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    dot = 0.0
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block
        dot += tl.sum(a_vals * b_vals)
    
    # Store result
    tl.store(output_ptr, dot)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    
    # Output tensor for the dot product result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    vdotr_kernel[(1,)](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()