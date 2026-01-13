import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block sum
    block_sum = 0.0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block
        block_dot = tl.sum(a_vals * b_vals, axis=0)
        block_sum += block_dot
    
    # Store the result
    tl.store(output_ptr, block_sum)

def s313_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block since we need global reduction
    grid = (1,)
    
    s313_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()