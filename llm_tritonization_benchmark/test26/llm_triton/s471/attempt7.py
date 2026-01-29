import triton
import triton.language as tl

@triton.jit
def s471_kernel(b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # x[i] = b[i] + d[i] * d[i] (using original b values)
    x_vals = b_vals + d_vals * d_vals
    tl.store(x_ptr + idx, x_vals, mask=mask)
    
    # b[i] = c[i] + d[i] * e[i] (update b after x computation)
    b_new = c_vals + d_vals * e_vals
    tl.store(b_ptr + idx, b_new, mask=mask)

def s471_triton(b, c, d, e, x):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s471_kernel[grid](b, c, d, e, x, n_elements, BLOCK_SIZE)