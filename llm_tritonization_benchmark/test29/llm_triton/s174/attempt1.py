import triton
import triton.language as tl

@triton.jit
def s174_kernel(a_ptr, b_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < m
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+M]
    store_indices = indices + m
    tl.store(a_ptr + store_indices, result, mask=mask)

def s174_triton(a, b, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s174_kernel[grid](
        a, b, m,
        BLOCK_SIZE=BLOCK_SIZE
    )