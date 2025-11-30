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
    # This kernel implements induction variable recognition
    # s starts at 0 and accumulates 2.0 for each iteration
    # a[i] = s * b[i] where s = 2.0 * (i + 1)
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Calculate s values: s = 2.0 * (i + 1)
    # Since s starts at 0 and adds 2.0 each iteration
    s_vals = 2.0 * (offsets + 1).to(tl.float32)
    
    # Calculate result: a[i] = s * b[i]
    result = s_vals * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s453_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )