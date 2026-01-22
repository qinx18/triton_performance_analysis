import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with the absolute value of a[0]
    first_val = tl.load(a_ptr)
    global_max = tl.abs(first_val)
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Update global maximum for each element in the block
        global_max = tl.maximum(global_max, tl.max(abs_vals))
    
    # Store the result
    tl.store(output_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s3113_kernel[grid](a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output.item()