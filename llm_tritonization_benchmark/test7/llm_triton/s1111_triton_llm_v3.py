import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Input offsets for b, c, d (stride 1)
    input_offsets = block_start + offsets
    # Output offsets for a (stride 2, writing to even indices)
    output_offsets = 2 * input_offsets
    
    # Mask for valid elements
    mask = input_offsets < n_elements
    
    # Load inputs
    b_vals = tl.load(b_ptr + input_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + input_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    # Store to a[2*i]
    tl.store(a_ptr + output_offsets, result, mask=mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)