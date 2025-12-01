import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        # Calculate current offsets
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices and direct values
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indirect values from b
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute result
        result = b_vals + c_vals
        
        # Store result to indirect locations in a
        tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s4113_kernel[(1,)](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a