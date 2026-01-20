import triton
import triton.language as tl

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Load indices
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Gather from b using indices
    gathered_values = tl.load(b_ptr + indices, mask=mask)
    
    # Store to a
    tl.store(a_ptr + offsets, gathered_values, mask=mask)

def vag_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vag_kernel[grid](a, b, ip, N, BLOCK_SIZE=BLOCK_SIZE)