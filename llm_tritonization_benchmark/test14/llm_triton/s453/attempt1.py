import torch
import triton
import triton.language as tl

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements induction variable recognition
    # s starts at 0 and increments by 2 for each i
    # a[i] = s * b[i] where s = 2 * (i + 1)
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Calculate s values: s = 2 * (i + 1)
    # For each position i, s has accumulated 2.0 exactly (i+1) times
    s_vals = 2.0 * (block_start + offsets + 1)
    
    # Calculate a[i] = s * b[i]
    a_vals = s_vals * b_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )