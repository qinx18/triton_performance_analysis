import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < (n - 1)
    
    # Load from read-only copy for a[i + k]
    read_indices = indices + k
    read_mask = mask & (read_indices < n)
    a_vals = tl.load(a_copy_ptr + read_indices, mask=read_mask, other=0.0)
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_copy, b, c, k, n,
        BLOCK_SIZE=BLOCK_SIZE
    )