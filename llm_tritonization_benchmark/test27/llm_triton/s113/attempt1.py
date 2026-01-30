import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for current block
    pid = tl.program_id(0)
    
    # Calculate block offsets (starting from index 1)
    block_start = pid * BLOCK_SIZE + 1
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Create mask for valid indices
    mask = offsets < N
    
    # Load a[0] (broadcast to all elements in block)
    a_0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a_0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # We process indices 1 to N-1, so total elements = N-1
    n_elements = N - 1
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s113_kernel[(grid_size,)](
        a, b, N, BLOCK_SIZE
    )