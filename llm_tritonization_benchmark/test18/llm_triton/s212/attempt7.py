import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block ID and calculate offsets
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for valid elements
    mask = idx < n_elements
    
    # Load current values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load a[i+1] values from copy
    next_idx = idx + 1
    next_mask = next_idx < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_idx, mask=next_mask, other=0.0)
    
    # Perform computations
    new_a = a_vals * c_vals
    new_b = b_vals + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy
    a_copy = a.clone()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )