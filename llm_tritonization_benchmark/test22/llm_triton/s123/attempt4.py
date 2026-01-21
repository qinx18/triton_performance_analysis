import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process multiple elements per program to improve efficiency
    start_i = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = start_i + offsets
    
    # Mask for valid elements
    mask = i_offsets < n_half
    
    # Load values for all elements in the block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate j indices (j starts at 0, 2, 4, ...)
    j_offsets = 2 * i_offsets
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    val1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets, val1, mask=mask)
    
    # Conditional assignment: if c[i] > 0, store at j+1
    cond_mask = (c_vals > 0.0) & mask
    val2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets + 1, val2, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )