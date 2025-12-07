import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load data
    a_vals = tl.load(a_copy_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load a[i+1] values for the second statement
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)
    a_plus_1_vals = tl.load(a_copy_ptr + idx_plus_1, mask=mask_plus_1)
    
    # First statement: a[i] *= c[i]
    new_a_vals = a_vals * c_vals
    
    # Second statement: b[i] += a[i + 1] * d[i]
    new_b_vals = b_vals + a_plus_1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d, a_copy, n_elements, BLOCK_SIZE
    )