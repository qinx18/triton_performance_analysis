import triton
import triton.language as tl
import torch

@triton.jit
def s221_first_kernel(a, c, d, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n) & (offsets >= 1)
    
    a_vals = tl.load(a + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    result = a_vals + c_vals * d_vals
    tl.store(a + offsets, result, mask=mask)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    # First statement: a[i] += c[i] * d[i] (parallelizable)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s221_first_kernel[grid](a, c, d, n, BLOCK_SIZE)
    
    # Second statement: b[i] = b[i-1] + a[i] + d[i] (sequential)
    # Use torch operations for the prefix sum pattern
    if n > 1:
        addends = a[1:] + d[1:]
        prefix_sums = torch.cumsum(addends, dim=0)
        b[1:] = b[0] + prefix_sums
    
    return a, b