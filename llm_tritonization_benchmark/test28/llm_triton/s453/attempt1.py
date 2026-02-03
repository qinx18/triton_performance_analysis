import torch
import triton
import triton.language as tl

@triton.jit
def s453_kernel(
    a_ptr, b_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute current offsets for this block
    current_offsets = block_start + offsets
    
    # Create mask for boundary checking
    mask = current_offsets < N
    
    # Load b values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute s values using closed form: s = 2 * (i + 1)
    s_vals = 2.0 * (current_offsets + 1).to(tl.float32)
    
    # Compute a[i] = s * b[i]
    a_vals = s_vals * b_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s453_kernel[grid](
        a, b,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a