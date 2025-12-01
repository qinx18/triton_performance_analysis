import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(
    a_ptr, b_ptr,
    n_elements,
    dot_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate dot product with unrolled computation
    offsets = tl.arange(0, BLOCK_SIZE)
    dot_acc = 0.0
    
    # Process elements in blocks of 5 (unrolled)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load blocks from both arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block
        dot_vals = a_vals * b_vals
        dot_acc += tl.sum(dot_vals, axis=0)
    
    # Store the final dot product result
    if tl.program_id(0) == 0:
        tl.store(dot_ptr, dot_acc)

def s352_triton(a, b):
    n_elements = a.numel()
    
    # Create output tensor for dot product result
    dot = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    grid = (1,)
    
    s352_kernel[grid](
        a, b,
        n_elements,
        dot,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dot.item()