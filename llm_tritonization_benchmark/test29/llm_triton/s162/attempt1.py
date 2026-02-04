import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx < n - 1) & (idx + k < n)
    
    # Load from read-only copy for a[i + k]
    a_shifted = tl.load(a_copy_ptr + idx + k, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute result
    result = a_shifted + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    s162_kernel[grid](a, a_copy, b, c, k, n, BLOCK_SIZE)