import triton
import triton.language as tl

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute s for each element: s = (i + 1) * 2
    s_vals = (offsets + 1) * 2.0
    
    # Compute a[i] = s * b[i]
    result = s_vals * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )