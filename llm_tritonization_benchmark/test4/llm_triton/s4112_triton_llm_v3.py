import torch
import triton
import triton.language as tl

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr, s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        # Calculate current offsets
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load values from a and b
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute and store result
        result = a_vals + b_vals * s
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s4112_kernel[(1,)](
        a, b, ip, s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a