import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy for a[i+1] and from b for b[i]
    a_read_indices = indices + 1
    a_read_mask = a_read_indices < (n + 1)
    
    a_vals = tl.load(a_copy_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # Only process LEN_1D - 1 elements
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, n, BLOCK_SIZE=BLOCK_SIZE
    )