import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Input indices for reading from b, c, d
    input_indices = block_start + offsets
    input_mask = input_indices < n_elements
    
    # Output indices for writing to a (2*i pattern)
    output_indices = 2 * input_indices
    output_mask = input_indices < n_elements
    
    # Load inputs
    b_vals = tl.load(b_ptr + input_indices, mask=input_mask)
    c_vals = tl.load(c_ptr + input_indices, mask=input_mask)
    d_vals = tl.load(d_ptr + input_indices, mask=input_mask)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    # Store to a[2*i]
    tl.store(a_ptr + output_indices, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)