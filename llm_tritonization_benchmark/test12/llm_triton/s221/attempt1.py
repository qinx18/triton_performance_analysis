import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one strip sequentially
    strip_id = tl.program_id(0)
    strip_start = strip_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 1  # +1 because loop starts from i=1
    
    mask = idx < n_elements + 1
    
    # Load values
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_prev_vals = tl.load(b_ptr + idx - 1, mask=mask)
    
    # Compute: a[i] += c[i] * d[i]
    new_a_vals = a_vals + c_vals * d_vals
    
    # Compute: b[i] = b[i-1] + a[i] + d[i]
    new_b_vals = b_prev_vals + new_a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)

def s221_triton(a, b, c, d):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1  # Loop from i=1 to LEN_1D-1
    
    BLOCK_SIZE = 1  # Must be 1 due to loop-carried dependency
    num_strips = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Process strips sequentially due to dependency
    for strip_id in range(num_strips):
        s221_kernel[(1,)](
            a, b, c, d,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )