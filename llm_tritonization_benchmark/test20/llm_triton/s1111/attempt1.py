import triton
import triton.language as tl
import torch

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for input arrays (0 to BLOCK_SIZE-1)
    offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = block_start + offsets
    
    # Create output offsets (2*i pattern)
    output_offsets = 2 * input_offsets
    
    # Create masks
    input_mask = input_offsets < n
    output_mask = output_offsets < (2 * n)
    
    # Load data
    b_vals = tl.load(b_ptr + input_offsets, mask=input_mask, other=0.0)
    c_vals = tl.load(c_ptr + input_offsets, mask=input_mask, other=0.0)
    d_vals = tl.load(d_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    term1 = c_vals * b_vals
    term2 = d_vals * b_vals
    term3 = c_vals * c_vals
    term4 = d_vals * b_vals
    term5 = d_vals * c_vals
    
    result = term1 + term2 + term3 + term4 + term5
    
    # Store to a[2*i]
    combined_mask = input_mask & output_mask
    tl.store(a_ptr + output_offsets, result, mask=combined_mask)

def s1111_triton(a, b, c, d):
    # Get array length from input tensors
    N = b.shape[0]
    n_elements = N // 2
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1111_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )