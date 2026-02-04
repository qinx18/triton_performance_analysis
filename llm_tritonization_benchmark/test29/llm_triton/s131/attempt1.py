import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy with offset
    a_vals = tl.load(a_copy_ptr + indices + m, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # Process only up to LEN_1D - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, m, n,
        BLOCK_SIZE=BLOCK_SIZE
    )