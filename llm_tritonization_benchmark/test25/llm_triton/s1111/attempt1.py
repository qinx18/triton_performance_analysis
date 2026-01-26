import triton
import triton.language as tl
import torch

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate block boundaries
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Input indices for reading from b, c, d
    input_indices = block_start + offsets
    input_mask = input_indices < n
    
    # Output indices for writing to a (stride of 2)
    output_indices = 2 * input_indices
    output_mask = input_indices < n
    
    # Load values from input arrays
    b_vals = tl.load(b_ptr + input_indices, mask=input_mask, other=0.0)
    c_vals = tl.load(c_ptr + input_indices, mask=input_mask, other=0.0)
    d_vals = tl.load(d_ptr + input_indices, mask=input_mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    # Store to a[2*i]
    tl.store(a_ptr + output_indices, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    # Get array size
    N = b.shape[0] // 2  # Loop goes from 0 to LEN_1D/2
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, N, BLOCK_SIZE)