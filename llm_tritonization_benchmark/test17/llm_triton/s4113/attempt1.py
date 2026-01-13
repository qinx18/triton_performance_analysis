import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices and c values
        indices = tl.load(ip_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Load b values using indirect addressing
        b_vals = tl.load(b_ptr + indices, mask=mask)
        
        # Compute result
        result = b_vals + c_vals
        
        # Store result using indirect addressing
        tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s4113_kernel[(1,)](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )