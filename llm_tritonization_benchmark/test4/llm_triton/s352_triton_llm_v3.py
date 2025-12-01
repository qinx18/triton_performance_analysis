import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize dot accumulator
    dot = 0.0
    
    # Process in blocks of 5 elements (unrolled dot product)
    for block_start in range(0, n, BLOCK_SIZE * 5):
        # Load 5 consecutive blocks
        for unroll_idx in range(5):
            current_offsets = block_start + unroll_idx * BLOCK_SIZE + offsets
            mask = current_offsets < n
            
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
            
            dot += tl.sum(a_vals * b_vals)
    
    # Store result (only first thread writes the final dot product)
    if pid == 0:
        tl.store(dot_ptr, dot)

def s352_triton(a, b):
    n = a.shape[0]
    
    # Create output tensor for dot product result
    dot_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 256
    grid = (1,)
    
    s352_kernel[grid](
        a, b, dot_result,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot_result.item()