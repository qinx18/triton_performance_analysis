import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block with stride inc
    for stride_offset in range(inc):
        # Current indices for this stride offset
        indices = block_start + offsets + stride_offset
        
        # Only process if index is valid and follows the stride pattern
        valid_mask = (indices < n - 1) & ((indices % inc) == (stride_offset % inc))
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:  # Only proceed if there are valid elements
            # Load a[i + inc] and b[i]
            a_read_indices = indices + inc
            a_read_mask = valid_mask & (a_read_indices < n)
            
            a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
            b_vals = tl.load(b_ptr + indices, mask=valid_mask, other=0.0)
            
            # Compute a[i] = a[i + inc] + b[i]
            result = a_vals + b_vals
            
            # Store back to a[i]
            tl.store(a_ptr + indices, result, mask=valid_mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Calculate how many elements will actually be processed
    num_elements = ((n - 1) + inc - 1) // inc * inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )