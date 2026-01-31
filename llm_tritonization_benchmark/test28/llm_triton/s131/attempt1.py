import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load from read-only copy for a[i + m] and from b for b[i]
    a_vals = tl.load(a_copy_ptr + idx + m, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # Loop goes to LEN_1D - 1
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, m, n, BLOCK_SIZE=BLOCK_SIZE
    )