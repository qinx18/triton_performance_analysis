import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    mask = (block_start + offsets) < n_elements
    
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)
    
    # Second pass: compute d[i] = a[i] + a[i+1]
    # We need to handle the dependency by loading a values after they've been computed
    mask_d = (block_start + offsets) < (n_elements - 1)
    
    a_i = tl.load(a_ptr + block_start + offsets, mask=mask_d)
    a_i_plus_1 = tl.load(a_ptr + block_start + offsets + 1, mask=mask_d)
    
    d_vals = a_i + a_i_plus_1
    tl.store(d_ptr + block_start + offsets, d_vals, mask=mask_d)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, b, c, d,
        n_elements - 1,  # For the d computation which goes to LEN_1D-1
        BLOCK_SIZE=BLOCK_SIZE
    )