import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Read from copy and b
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original a
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE
    )