import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k]
    tl.store(a_ptr + k + block_start + offsets, result, mask=mask)

def s173_triton(a, b):
    k = len(a) // 2
    n_elements = k
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s173_kernel[grid](
        a, b, n_elements, k, BLOCK_SIZE=BLOCK_SIZE
    )