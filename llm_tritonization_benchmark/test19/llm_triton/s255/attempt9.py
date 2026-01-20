import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load x and y values (scalar loads)
    x_ptr = b_ptr + (n - 1)
    y_ptr = b_ptr + (n - 2)
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x + y) * 0.333
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s255_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s255_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a

def s255_c(a, b, x):
    n = a.shape[0]
    y = b[n-2].item()
    
    for i in range(n):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i].item()
    
    return a