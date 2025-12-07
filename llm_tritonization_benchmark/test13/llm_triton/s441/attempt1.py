import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_val = tl.load(a_ptr + idx, mask=mask)
    b_val = tl.load(b_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    
    # Conditional computation
    cond1 = d_val < 0.0  # d[i] < 0
    cond2 = d_val == 0.0  # d[i] == 0
    
    # Calculate updates for each condition
    update1 = b_val * c_val  # b[i] * c[i]
    update2 = b_val * b_val  # b[i] * b[i]
    update3 = c_val * c_val  # c[i] * c[i]
    
    # Select appropriate update based on conditions
    result = tl.where(cond1, a_val + update1,
                     tl.where(cond2, a_val + update2, a_val + update3))
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )