import torch
import triton
import triton.language as tl

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    m,
    LEN_1D,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < LEN_1D - 1) & (offsets + m < LEN_1D)
    
    # Read from copy and b
    a_vals = tl.load(a_copy_ptr + offsets + m, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Write to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s131_triton(a, b, m):
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = a.size(0)
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_1D - 1, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        m,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a