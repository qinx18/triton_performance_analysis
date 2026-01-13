import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute dot product contribution
    products = a_vals * b_vals
    
    # Sum within this block
    block_sum = tl.sum(products)
    
    # Store the partial sum (each block contributes one value)
    if tl.program_id(0) == 0:
        tl.atomic_add(dot_ptr, block_sum)
    else:
        tl.atomic_add(dot_ptr, block_sum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Initialize result tensor
    dot_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Calculate grid and block sizes
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s352_kernel[(grid_size,)](
        a, b, dot_result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot_result.item()