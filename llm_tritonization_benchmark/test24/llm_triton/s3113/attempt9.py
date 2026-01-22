import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with the absolute value of a[0]
    first_val = tl.load(a_ptr, mask=True)
    block_max = tl.abs(first_val)
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Find maximum in this block among valid elements
        masked_abs_vals = tl.where(mask, abs_vals, -1.0)
        current_max = tl.max(masked_abs_vals)
        
        # Update global maximum only if we have valid elements
        has_valid = tl.sum(mask.to(tl.int32)) > 0
        if has_valid:
            if current_max > block_max:
                block_max = current_max
    
    # Store the result
    tl.store(output_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s3113_kernel[grid](a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output.item()