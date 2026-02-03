import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute s using closed form: s = 2*(i+1)
    s_vals = 2.0 * (idx + 1)
    
    # Compute a[i] = s * b[i]
    result = s_vals * b_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE,
    )