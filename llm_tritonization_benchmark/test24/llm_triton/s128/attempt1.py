import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Compute k values: k = 2*i
    k_offsets = 2 * i_offsets
    
    # Load values
    b_vals = tl.load(b_ptr + k_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store b[k]
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    n = a.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n_half, BLOCK_SIZE=BLOCK_SIZE)