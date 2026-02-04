import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start * inc + offsets * inc
    
    mask = indices < (n - 1)
    
    # Load from read-only copy and original b array
    a_read_indices = indices + inc
    read_mask = a_read_indices < n
    
    a_vals = tl.load(a_copy_ptr + a_read_indices, mask=mask & read_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of iterations (elements processed)
    num_elements = (n - 1 + inc - 1) // inc  # Ceiling division
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE
    )