import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Load a[0] once (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Process elements in blocks starting from index 1
    pid = tl.program_id(0)
    block_start = 1 + pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # Process elements from index 1 to N-1
    num_elements = N - 1
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, N, BLOCK_SIZE)