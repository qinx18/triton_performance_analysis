import triton
import triton.language as tl

@triton.jit
def s241_kernel1(a, b, c, d, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    result = b_vals * c_vals * d_vals
    tl.store(a + offsets, result, mask=mask)

@triton.jit
def s241_kernel2(a, b, d, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a + offsets, mask=mask)
    a_next_vals = tl.load(a + offsets + 1, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    result = a_vals * a_next_vals * d_vals
    tl.store(b + offsets, result, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel1[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    s241_kernel2[grid](a, b, d, n_elements, BLOCK_SIZE)