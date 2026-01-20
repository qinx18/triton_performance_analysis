import triton
import triton.language as tl

@triton.jit
def s1279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    cond1 = a_vals < 0.0
    cond2 = b_vals > a_vals
    combined_cond = cond1 & cond2
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    update = d_vals * e_vals
    c_vals = tl.where(combined_cond, c_vals + update, c_vals)
    
    tl.store(c_ptr + offsets, c_vals, mask=mask)

def s1279_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1279_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)