import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes sum reduction over specific elements of array a
    # Each block handles one iteration of the original loop
    
    # Single block computes the entire sum
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Compute sum += test(a) - sum first 4 elements starting at a[0]
    offsets = tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[4]) - sum 4 elements starting at a[4]
    offsets = 4 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[8]) - sum 4 elements starting at a[8]
    offsets = 8 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[12]) - sum 4 elements starting at a[12]
    offsets = 12 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[16]) - sum 4 elements starting at a[16]
    offsets = 16 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[20]) - sum 4 elements starting at a[20]
    offsets = 20 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[24]) - sum 4 elements starting at a[24]
    offsets = 24 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Compute sum += test(&a[28]) - sum 4 elements starting at a[28]
    offsets = 28 + tl.arange(0, 4)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 32
    
    # Launch single block to compute the sum
    grid = (1,)
    
    s31111_kernel[grid](
        a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a