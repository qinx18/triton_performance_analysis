import triton
import triton.language as tl

@triton.jit
def s442_kernel(a, b, c, d, e, indx, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    a_vals = tl.load(a + offsets, mask=mask)
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    e_vals = tl.load(e + offsets, mask=mask)
    indx_vals = tl.load(indx + offsets, mask=mask)
    
    # Compute updates based on index values
    case1 = indx_vals == 1
    case2 = indx_vals == 2  
    case3 = indx_vals == 3
    case4 = indx_vals == 4
    
    update_vals = tl.where(case1, b_vals * b_vals,
                  tl.where(case2, c_vals * c_vals,
                  tl.where(case3, d_vals * d_vals, 
                           e_vals * e_vals)))
    
    result = a_vals + update_vals
    
    tl.store(a + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](a, b, c, d, e, indx, N, BLOCK_SIZE)