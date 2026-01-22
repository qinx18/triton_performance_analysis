import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, j: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the element index for this row
    i = 1 + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = i < N
    
    # Calculate linear indices
    current_idx = j * N + i
    left_idx = j * N + (i - 1)
    up_idx = (j - 1) * N + i
    
    # Load values
    left_val = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
    up_val = tl.load(aa_ptr + up_idx, mask=mask, other=0.0)
    
    # Compute new value
    new_val = (left_val + up_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_idx, new_val, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process rows sequentially
    for j in range(1, N):
        num_elements = N - 1
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, 
            N=N, 
            j=j, 
            BLOCK_SIZE=BLOCK_SIZE
        )