import torch
import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, b_ptr, output_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from a[i+k] and b[i]
    a_offsets = offsets + k
    a_mask = (offsets < n_elements) & (a_offsets < n_elements + k)
    b_mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=b_mask, other=0.0)
    
    # Compute a[i] = a[i+k] + b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def s431_triton(a, b):
    k1 = 1
    k2 = 2
    k = 2 * k1 - k2  # k = 0
    
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.empty_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, b, output,
        n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Copy result back to a
    a.copy_(output)
    
    return a