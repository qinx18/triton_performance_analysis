import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_input, BLOCK_SIZE: tl.constexpr):
    # Each program handles one input element
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_input
    
    # Load input data
    b_vals = tl.load(b_ptr + i, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i, mask=mask, other=0.0)
    
    # Compute de = d[i] * e[i]
    de_vals = d_vals * e_vals
    
    # First output: a[j] where j starts from i (since j++ happens before first assignment)
    out1_vals = b_vals + de_vals
    j1_positions = 2 * i  # j increments from -1, so at i=0, j=0, i=1, j>=2, etc.
    
    # Store first output where valid
    tl.store(a_ptr + j1_positions, out1_vals, mask=mask)
    
    # Second output: conditionally written at next j position
    c_positive = c_vals > 0.0
    out2_vals = c_vals + de_vals
    j2_positions = 2 * i + 1
    out2_mask = mask & c_positive
    
    # Store second output only where condition is met and within bounds
    tl.store(a_ptr + j2_positions, out2_vals, mask=out2_mask)

def s123_triton(a, b, c, d, e):
    n_input = b.shape[0] // 2  # Loop goes to LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_input, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_input,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a