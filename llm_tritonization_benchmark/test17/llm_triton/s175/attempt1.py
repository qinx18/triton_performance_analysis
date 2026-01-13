import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + idx + inc, mask=mask & ((idx + inc) < (n + 1)))
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s175_triton(a, b, inc):
    n = len(a) - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Launch kernel
    s175_kernel[(grid_size,)](
        a, a_copy, b, n, inc, BLOCK_SIZE
    )