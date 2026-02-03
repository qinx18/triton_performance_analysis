import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx >= 1) & (idx < n - 1)
    
    # First statement: a[i] = b[i-1] + c[i]
    b_prev_idx = idx - 1
    b_prev_mask = mask & (b_prev_idx >= 0)
    b_prev = tl.load(b_ptr + b_prev_idx, mask=b_prev_mask, other=0.0)
    c_curr = tl.load(c_ptr + idx, mask=mask, other=0.0)
    
    a_new = b_prev + c_curr
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # Second statement: b[i] = a[i+1] * d[i] (using original a values)
    a_next_idx = idx + 1
    a_next_mask = mask & (a_next_idx < n)
    a_next = tl.load(a_copy_ptr + a_next_idx, mask=a_next_mask, other=0.0)
    d_curr = tl.load(d_ptr + idx, mask=mask, other=0.0)
    
    b_new = a_next * d_curr
    tl.store(b_ptr + idx, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1213_kernel[grid](a, b, c, d, a_copy, n, BLOCK_SIZE=BLOCK_SIZE)