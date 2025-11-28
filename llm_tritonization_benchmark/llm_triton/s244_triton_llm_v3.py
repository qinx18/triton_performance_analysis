import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second statement: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)
    
    # Third statement: a[i+1] = b[i] + a[i+1] * d[i]
    # Need to handle the offset carefully
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)  # Allow one extra element
    
    # Load a[i+1] values
    a_plus1_vals = tl.load(a_ptr + offsets_plus1, mask=mask_plus1)
    
    # Compute new values: b[i] + a[i+1] * d[i]
    a_plus1_new = b_new + a_plus1_vals * d_vals
    
    # Store back to a[i+1]
    tl.store(a_ptr + offsets_plus1, a_plus1_new, mask=mask_plus1)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )