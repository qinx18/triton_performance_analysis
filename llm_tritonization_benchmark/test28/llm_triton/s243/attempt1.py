import triton
import triton.language as tl

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Load a[i+1] values from read-only copy
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n + 1)  # Allow reading a[i+1]
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_vals = a_vals + d_vals * e_vals
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_vals = b_vals + a_next_vals * d_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e, n, BLOCK_SIZE
    )