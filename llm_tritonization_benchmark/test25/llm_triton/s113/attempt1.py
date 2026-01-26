import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start (skip index 0)
    block_start = 1 + pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask to handle boundaries
    mask = current_offsets < N
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # We process from index 1 to N-1, so we have N-1 elements to process
    num_elements = N - 1
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    grid = (num_blocks,)
    
    s113_kernel[grid](a, b, N, BLOCK_SIZE)