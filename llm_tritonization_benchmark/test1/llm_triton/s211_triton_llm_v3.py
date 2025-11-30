import triton
import triton.language as tl

@triton.jit
def s211_kernel(
    a_ptr,
    b_ptr,
    b_copy_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements - 1
    
    # Load from read-only copy for b[i-1] and b[i+1]
    b_prev = tl.load(b_copy_ptr + offsets - 1, mask=mask)
    b_next = tl.load(b_copy_ptr + offsets + 1, mask=mask)
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute results
    a_result = b_prev + c_vals * d_vals
    b_result = b_next - e_vals * d_vals
    
    # Store to original arrays
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of b array
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = ((n_elements - 2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s211_kernel[grid](
        a,
        b,
        b_copy,
        c,
        d,
        e,
        n_elements,
        BLOCK_SIZE,
    )