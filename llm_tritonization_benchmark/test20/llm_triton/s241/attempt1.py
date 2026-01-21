import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (n - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First statement: a[i] = b[i] * c[i] * d[i]
    a_new = b_vals * c_vals * d_vals
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Second statement: b[i] = a[i] * a[i+1] * d[i]
    # Use read-only copy for a[i] and a[i+1] reads
    a_copy_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    a_copy_next_vals = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    
    b_new = a_new * a_copy_next_vals * d_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    n = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )