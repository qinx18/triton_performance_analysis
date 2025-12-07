import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load original values from read-only copy
    a_i = tl.load(a_copy_ptr + offsets, mask=mask, other=0.0)
    a_i1 = tl.load(a_copy_ptr + offsets + 1, mask=mask, other=0.0)
    a_i2 = tl.load(a_copy_ptr + offsets + 2, mask=mask, other=0.0)
    a_i3 = tl.load(a_copy_ptr + offsets + 3, mask=mask, other=0.0)
    a_i4 = tl.load(a_copy_ptr + offsets + 4, mask=mask, other=0.0)
    a_i5 = tl.load(a_copy_ptr + offsets + 5, mask=mask, other=0.0)
    
    # Check which positions to update based on (i % 5)
    mod5 = offsets % 5
    
    # Compute results for each case using original values
    result = tl.where(mod5 == 0, a_i1 * a_i,
             tl.where(mod5 == 1, a_i2 * a_i1,  
             tl.where(mod5 == 2, a_i3 * a_i2,
             tl.where(mod5 == 3, a_i4 * a_i3,
                      a_i5 * a_i4))))  # mod5 == 4
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s116_triton(a):
    n_elements = a.shape[0] - 5
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )