import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load b values for the current block
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Handle x values (previous b values with wrapping)
    x_offsets = (block_start + offsets - 1) % n_elements
    x_vals = tl.load(b_ptr + x_offsets, mask=mask, other=0.0)
    
    # For i=0, x should be b[n_elements-1]
    first_elem_mask = (block_start + offsets) == 0
    last_b_val = tl.load(b_ptr + n_elements - 1)
    x_vals = tl.where(first_elem_mask, last_b_val, x_vals)
    
    # Handle y values (b values from i-2 with wrapping)
    y_offsets = (block_start + offsets - 2) % n_elements
    y_vals = tl.load(b_ptr + y_offsets, mask=mask, other=0.0)
    
    # For i=0, y should be b[n_elements-2]
    # For i=1, y should be b[n_elements-1]
    second_last_b_val = tl.load(b_ptr + n_elements - 2)
    y_vals = tl.where(first_elem_mask, second_last_b_val, y_vals)
    second_elem_mask = (block_start + offsets) == 1
    y_vals = tl.where(second_elem_mask, last_b_val, y_vals)
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x_vals + y_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s255_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )