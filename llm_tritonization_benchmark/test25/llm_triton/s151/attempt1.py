import triton
import triton.language as tl

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load a[i+m] and b[i]
    a_shifted_offsets = idx + m
    a_shifted_mask = mask & (a_shifted_offsets < n + m)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_shifted_vals = tl.load(a_ptr + a_shifted_offsets, mask=a_shifted_mask)
    
    # Compute a[i] = a[i+m] + b[i]
    result = a_shifted_vals + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s151_triton(a, b):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    m = 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n, m, BLOCK_SIZE)
    
    return a