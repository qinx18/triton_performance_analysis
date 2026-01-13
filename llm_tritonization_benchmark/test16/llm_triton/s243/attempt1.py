import triton
import triton.language as tl

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < (n - 1)
    current_offsets = block_start + offsets
    next_offsets = block_start + offsets + 1
    
    # Load values for first statement: a[i] = b[i] + c[i] * d[i]
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # First statement
    a_new = b_vals + c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    b_new = a_new + d_vals * e_vals
    tl.store(b_ptr + current_offsets, b_new, mask=mask)
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    # Use read-only copy for a[i+1] to avoid race condition
    next_mask = (block_start + offsets + 1) < n
    a_next = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    a_final = b_new + a_next * d_vals
    tl.store(a_ptr + current_offsets, a_final, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e, n, BLOCK_SIZE
    )