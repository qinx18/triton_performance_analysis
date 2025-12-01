import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets + 1  # i ranges from 1 to LEN_1D-2
    
    mask = idx < n_elements - 1
    
    # Load values for a[i] = b[i-1] + c[i] * d[i]
    b_prev = tl.load(b_copy_ptr + idx - 1, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load values for b[i] = b[i+1] - e[i] * d[i]
    b_next = tl.load(b_copy_ptr + idx + 1, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute results
    a_result = b_prev + c_vals * d_vals
    b_result = b_next - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_result, mask=mask)
    tl.store(b_ptr + idx, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = ((n_elements - 2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )