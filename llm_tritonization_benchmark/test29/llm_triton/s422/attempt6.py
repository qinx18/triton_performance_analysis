import triton
import triton.language as tl

@triton.jit
def s422_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    flat_vals = tl.load(flat_2d_array_ptr + offsets + 8, mask=mask)
    result = flat_vals + a_vals
    
    tl.store(xx_ptr + offsets, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        a, flat_2d_array, xx, n_elements, BLOCK_SIZE
    )