import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute conditions
    d_negative = d_vals < 0.0
    d_zero = d_vals == 0.0
    
    # Compute updates based on conditions
    update1 = b_vals * c_vals  # if d[i] < 0
    update2 = b_vals * b_vals  # if d[i] == 0
    update3 = c_vals * c_vals  # else (d[i] > 0)
    
    # Select appropriate update
    result = tl.where(d_negative, update1, tl.where(d_zero, update2, update3))
    
    # Update a[i]
    new_a = a_vals + result
    
    # Store result
    tl.store(a_ptr + indices, new_a, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements, BLOCK_SIZE
    )