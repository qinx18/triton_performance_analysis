import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices for this block
    indices = block_start + offsets
    mask = indices < (n - 1)
    next_indices = indices + 1
    next_mask = next_indices < n
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    c_squared = c_vals * c_vals
    b_squared = b_vals * b_vals
    a_vals = b_vals + c_squared + b_squared + c_vals
    
    # Store a[i]
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Load a[i+1] from the read-only copy
    a_next_vals = tl.load(a_copy_ptr + next_indices, mask=next_mask, other=0.0)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_next_vals
    
    # Store d[i]
    tl.store(d_ptr + indices, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy of a to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )