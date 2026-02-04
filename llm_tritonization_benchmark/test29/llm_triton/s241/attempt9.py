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
def s241_kernel2(a, a_copy, b, d, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Check bounds for a[i+1] access
    next_mask = (offsets + 1) < (n_elements + 1)
    combined_mask = mask & next_mask
    
    a_vals = tl.load(a_copy + offsets, mask=combined_mask)
    a_next_vals = tl.load(a_copy + offsets + 1, mask=combined_mask)
    d_vals = tl.load(d + offsets, mask=combined_mask)
    
    result = a_vals * a_next_vals * d_vals
    tl.store(b + offsets, result, mask=combined_mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel1[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    a_copy = a.clone()
    s241_kernel2[grid](a, a_copy, b, d, n_elements, BLOCK_SIZE)