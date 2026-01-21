import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Load indices and values
    indices = tl.load(ip_ptr + offsets, mask=mask)
    values = tl.load(b_ptr + offsets, mask=mask)
    
    # Scatter values to a using indices
    tl.store(a_ptr + indices, values, mask=mask)

def vas_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vas_kernel[grid](
        a, b, ip,
        N, BLOCK_SIZE=BLOCK_SIZE
    )