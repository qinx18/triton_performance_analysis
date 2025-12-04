import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Input offsets for b, c, d arrays
    input_offsets = block_start + offsets
    input_mask = input_offsets < n_elements
    
    # Output offsets for a array (2*i indexing)
    output_offsets = 2 * input_offsets
    
    # Load input values
    b_vals = tl.load(b_ptr + input_offsets, mask=input_mask, other=0.0)
    c_vals = tl.load(c_ptr + input_offsets, mask=input_mask, other=0.0)
    d_vals = tl.load(d_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    # Store result at a[2*i]
    tl.store(a_ptr + output_offsets, result, mask=input_mask)

def s1111_triton(a, b, c, d):
    n_elements = b.numel() // 2  # LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    return a