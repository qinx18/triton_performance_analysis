import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy for a[i+k]
    a_copy_offsets = indices + k
    a_copy_mask = a_copy_offsets < n
    a_copy_vals = tl.load(a_copy_ptr + a_copy_offsets, mask=a_copy_mask, other=0.0)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_copy_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b, k):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s431_kernel[grid](a, a_copy, b, k, n, BLOCK_SIZE)