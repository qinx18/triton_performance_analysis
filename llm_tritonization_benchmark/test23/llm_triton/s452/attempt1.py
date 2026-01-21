import triton
import triton.language as tl

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Convert offsets to float and add 1 for (i+1) term
    i_plus_1 = (offsets + 1).to(tl.float32)
    
    result = b_vals + c_vals * i_plus_1
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s452_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s452_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)