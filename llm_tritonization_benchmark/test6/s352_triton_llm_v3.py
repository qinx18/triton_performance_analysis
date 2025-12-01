import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize dot product accumulator
    dot = 0.0
    
    # Process elements in blocks of 5
    for block_start in range(0, n_elements, BLOCK_SIZE * 5):
        # Process 5 consecutive elements per iteration
        for j in range(5):
            current_offsets = block_start + j * BLOCK_SIZE + offsets
            mask = current_offsets < n_elements
            
            # Load values
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
            
            # Accumulate dot product
            dot += tl.sum(a_vals * b_vals)
    
    # Store result (only first thread writes)
    if pid == 0:
        tl.store(result_ptr, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Prepare result tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single block
    BLOCK_SIZE = 256
    grid = (1,)
    
    s352_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()