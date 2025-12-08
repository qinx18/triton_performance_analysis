import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_copy_vals = tl.load(a_copy_ptr + idx, mask=mask)
    
    # Load a[i+1] values with bounds checking
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)  # Allow loading one past valid range
    a_copy_plus1 = tl.load(a_copy_ptr + idx_plus1, mask=mask_plus1, other=0.0)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    a_new = b_vals * c_vals * d_vals
    
    # Store new a values
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    b_new = a_new * a_copy_plus1 * d_vals
    
    # Store new b values
    tl.store(b_ptr + idx, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )