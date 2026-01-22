import triton
import triton.language as tl

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load indices
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Store to a
    tl.store(a_ptr + offsets, b_vals, mask=mask)

def vag_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vag_kernel[grid](a, b, ip, n, BLOCK_SIZE=BLOCK_SIZE)