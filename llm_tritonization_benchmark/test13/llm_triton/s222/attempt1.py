import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * 1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = (idx < n_elements) & (idx >= 0)
    
    # Load values for first operation: a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + idx + 1, mask=mask)
    b_vals = tl.load(b_ptr + idx + 1, mask=mask)
    c_vals = tl.load(c_ptr + idx + 1, mask=mask)
    
    # First operation
    a_vals = a_vals + b_vals * c_vals
    
    # Load e values for recurrence: e[i] = e[i-1] * e[i-1]
    e_prev = tl.load(e_ptr + idx, mask=mask)  # e[i-1]
    e_new = e_prev * e_prev
    
    # Third operation: a[i] -= b[i] * c[i]
    a_vals = a_vals - b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + idx + 1, a_vals, mask=mask)
    tl.store(e_ptr + idx + 1, e_new, mask=mask)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0] - 1  # Loop from 1 to LEN_1D-1
    BLOCK_SIZE = 1
    
    num_strips = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Process strips sequentially due to loop-carried dependency
    for strip_id in range(num_strips):
        remaining = min(BLOCK_SIZE, n_elements - strip_id * BLOCK_SIZE)
        if remaining <= 0:
            break
            
        s222_kernel[(1,)](
            a, b, c, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )