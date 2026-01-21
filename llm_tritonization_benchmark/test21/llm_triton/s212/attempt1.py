import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load values
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load a[i+1] values for second statement
    indices_plus1 = indices + 1
    mask_plus1 = indices_plus1 < (n + 1)  # Allow reading a[i+1] up to a[n]
    a_plus1_vals = tl.load(a_copy_ptr + indices_plus1, mask=mask_plus1)
    
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # First statement: a[i] *= c[i]
    new_a_vals = a_vals * c_vals
    
    # Second statement: b[i] += a[i + 1] * d[i]
    new_b_vals = b_vals + a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, new_a_vals, mask=mask)
    tl.store(b_ptr + indices, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    N = a.shape[0]
    n = N - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )