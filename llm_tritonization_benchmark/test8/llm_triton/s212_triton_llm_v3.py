import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Create temporary storage for a values
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load current a and c values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # First statement: a[i] *= c[i]
    a_new = a_vals * c_vals
    
    # Store updated a values
    tl.store(a_ptr + block_start + offsets, a_new, mask=mask)
    
    # Load b values and d values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    
    # Load a[i+1] values for second statement
    # Need to handle boundary condition for last element
    offsets_plus1 = offsets + 1
    mask_plus1 = (block_start + offsets_plus1) < (n_elements + 1)
    a_plus1_vals = tl.load(a_ptr + block_start + offsets_plus1, mask=mask_plus1)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    # Only update where we have valid a[i+1] values (i < n_elements-1)
    valid_mask = mask & ((block_start + offsets) < (n_elements - 1))
    b_update = tl.where(valid_mask, b_vals + a_plus1_vals * d_vals, b_vals)
    
    # Store updated b values
    tl.store(b_ptr + block_start + offsets, b_update, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )