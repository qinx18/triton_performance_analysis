import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load input arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Implement the conditional logic
    # if (d[i] < 0.) a[i] += b[i] * c[i];
    # else if (d[i] == 0.) a[i] += b[i] * b[i];  
    # else a[i] += c[i] * c[i];
    
    cond1 = d_vals < 0.0
    cond2 = d_vals == 0.0
    
    # Calculate the three possible additions
    add1 = b_vals * c_vals  # for d[i] < 0
    add2 = b_vals * b_vals  # for d[i] == 0
    add3 = c_vals * c_vals  # for d[i] > 0
    
    # Select the appropriate addition based on conditions
    result = tl.where(cond1, a_vals + add1, 
                     tl.where(cond2, a_vals + add2, a_vals + add3))
    
    # Store the result
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )