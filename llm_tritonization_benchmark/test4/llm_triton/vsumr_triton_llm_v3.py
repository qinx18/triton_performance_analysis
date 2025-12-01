import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum to zero
    sum_val = 0.0
    
    # Process data in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load data with masking
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val += tl.sum(a_vals)
    
    # Store result (only one thread stores the final sum)
    if tl.program_id(0) == 0:
        tl.store(a_ptr + n, sum_val)

def vsumr_triton(a):
    n = a.shape[0]
    
    # Extend tensor to store result
    extended_a = torch.cat([a, torch.zeros(1, device=a.device, dtype=a.dtype)])
    
    # Launch kernel with single program
    grid = (1,)
    BLOCK_SIZE = 1024
    
    vsumr_kernel[grid](
        extended_a,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the sum
    return extended_a[n].item()