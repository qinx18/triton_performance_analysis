import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Calculate reverse indices for a[LEN_1D-i-1]
    reverse_idx = n_elements - 1 - idx
    reverse_mask = mask
    
    # Load a[LEN_1D-i-1]
    a_reverse_vals = tl.load(a_ptr + reverse_idx, mask=reverse_mask)
    
    # Calculate x = a[LEN_1D-i-1] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store results
    # a[i] = x - 1.0
    tl.store(a_ptr + idx, x_vals - 1.0, mask=mask)
    # b[i] = x
    tl.store(b_ptr + idx, x_vals, mask=mask)
    # Store x for output
    tl.store(x_ptr + idx, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s281_kernel[grid](
        a, b, c, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )