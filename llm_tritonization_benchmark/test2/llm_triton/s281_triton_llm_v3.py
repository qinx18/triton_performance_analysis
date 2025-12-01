import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    # For reverse access: a[LEN_1D-i-1] becomes a[n_elements-1-i]
    reverse_offsets = n_elements - 1 - offsets
    reverse_mask = reverse_offsets >= 0
    combined_mask = mask & reverse_mask
    
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_reverse + b_vals * c_vals
    
    # Store results
    # a[i] = x - 1.0
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    # b[i] = x
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s281_kernel[(grid_size,)](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b