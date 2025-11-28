import triton
import triton.language as tl
import torch

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run with a single block since there are dependencies
    # between iterations that prevent parallelization
    pid = tl.program_id(axis=0)
    
    # Only process if this is the first (and only) block
    if pid != 0:
        return
    
    # Initialize scalar
    s = 0.0
    
    # Process elements sequentially
    for i in range(0, n_elements, BLOCK_SIZE):
        # Calculate how many elements to process in this iteration
        remaining = n_elements - i
        block_size = min(BLOCK_SIZE, remaining)
        
        # Create offset array for this block
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_size
        
        # Load elements
        e_vals = tl.load(e_ptr + i + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + i + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + i + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + i + offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for j in range(block_size):
            # a[i] = s*e[i]
            a_val = s * e_vals[j]
            tl.store(a_ptr + i + j, a_val)
            
            # s = b[i]+c[i]
            s = b_vals[j] + c_vals[j]
            
            # b[i] = a[i]+d[i]
            b_val = a_val + d_vals[j]
            tl.store(b_ptr + i + j, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Use a single block with sequential processing
    grid = lambda meta: (1,)
    BLOCK_SIZE = 1024
    
    s2251_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )