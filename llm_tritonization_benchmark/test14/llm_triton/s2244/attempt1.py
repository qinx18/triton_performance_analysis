import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get thread block offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Mask for valid elements (i < n_elements - 1)
    mask = current_offsets < (n_elements - 1)
    
    # Load data for current block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute values
    val_for_a_i_plus_1 = b_vals + e_vals  # a[i+1] = b[i] + e[i]
    val_for_a_i = b_vals + c_vals          # a[i] = b[i] + c[i]
    
    # Store a[i] = b[i] + c[i] for all valid positions
    tl.store(a_ptr + current_offsets, val_for_a_i, mask=mask)
    
    # Store a[i+1] = b[i] + e[i] for valid positions (i+1 < n_elements)
    i_plus_1_offsets = current_offsets + 1
    i_plus_1_mask = mask & (i_plus_1_offsets < n_elements)
    tl.store(a_ptr + i_plus_1_offsets, val_for_a_i_plus_1, mask=i_plus_1_mask)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for processing elements 0 to n_elements-2
    num_blocks = triton.cdiv(n_elements - 1, BLOCK_SIZE)
    
    # Launch kernel
    s2244_kernel[(num_blocks,)](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )