import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # First assignment: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # Load a_copy values for second assignment
    a_copy_vals = tl.load(a_copy_ptr + idx, mask=mask)
    
    # Load a[i+1] from copy (need to handle boundary)
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)
    a_copy_vals_plus_1 = tl.load(a_copy_ptr + idx_plus_1, mask=mask_plus_1)
    
    # Second assignment: b[i] = a[i] * a[i+1] * d[i]
    # Use the newly computed a values and the a_copy values for a[i+1]
    b_new_vals = a_vals * a_copy_vals_plus_1 * d_vals
    tl.store(b_ptr + idx, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )