import triton
import triton.language as tl

@triton.jit
def s322_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE + 2
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from read-only copy for dependencies
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    a_prev1 = tl.load(a_copy_ptr + offsets - 1, mask=mask)
    a_prev2 = tl.load(a_copy_ptr + offsets - 2, mask=mask)
    
    # Load other arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute result
    result = a_vals + a_prev1 * b_vals + a_prev2 * c_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements - 2, meta['BLOCK_SIZE']),)
    
    s322_kernel[grid](
        a,
        a_copy,
        b,
        c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )