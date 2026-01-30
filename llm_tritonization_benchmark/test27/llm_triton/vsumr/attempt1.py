import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block sum
    block_sum = 0.0
    
    # Process all elements in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load values with masking
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Accumulate sum for this block
        block_sum += tl.sum(vals, axis=0)
    
    # Store the result (only the first thread in the first block)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, block_sum)

def vsumr_triton(a):
    N = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use single program since we need global reduction
    BLOCK_SIZE = 1024
    grid = (1,)
    
    # Launch kernel
    vsumr_kernel[grid](
        a, output, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()