import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load b and c values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a values from reverse indices
    reverse_indices = n - 1 - indices
    reverse_mask = reverse_indices >= 0
    final_mask = mask & reverse_mask
    
    a_reverse_vals = tl.load(a_ptr + reverse_indices, mask=final_mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    a_new_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_new_vals, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    s281_kernel[(grid_size,)](a, b, c, n, BLOCK_SIZE)