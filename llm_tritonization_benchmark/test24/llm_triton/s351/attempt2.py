import triton
import triton.language as tl

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    result = a_vals + alpha * b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s351_triton(a, b, c):
    n = a.shape[0]
    alpha = c[0].item()
    
    # Ensure n is divisible by 5 for the unrolled loop pattern
    effective_n = (n // 5) * 5
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(effective_n, BLOCK_SIZE),)
    
    s351_kernel[grid](a, b, alpha, effective_n, BLOCK_SIZE=BLOCK_SIZE)