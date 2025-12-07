import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, a_0_val, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load b[i+1] values (since we're processing from index 1)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i+1] = a[0] + b[i+1]
    result = a_0_val + b_vals
    
    # Store result to a[i+1]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0] - 1  # Process from index 1 to end
    
    if n_elements <= 0:
        return
    
    # Get a[0] value as scalar
    a_0_val = a[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s113_kernel[grid](
        a[1:],  # Start from index 1
        b[1:],  # Start from index 1
        a_0_val,  # Pass a[0] as scalar
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )