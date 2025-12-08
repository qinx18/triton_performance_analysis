import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load current elements
    a_vals = tl.load(a_copy_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load a[i+1] elements (from read-only copy)
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)
    a_plus1_vals = tl.load(a_copy_ptr + idx_plus1, mask=mask_plus1)
    
    # Compute: a[i] *= c[i]
    new_a = a_vals * c_vals
    
    # Compute: b[i] += a[i + 1] * d[i]
    new_b = b_vals + a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d, n_elements, BLOCK_SIZE
    )