import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, j: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1
    mask = i_indices < N
    
    # Load values
    left_offsets = j * N + (i_indices - 1)  # aa[j][i-1]
    up_offsets = (j - 1) * N + i_indices    # aa[j-1][i]
    
    left_vals = tl.load(aa_ptr + left_offsets, mask=mask, other=0.0)
    up_vals = tl.load(aa_ptr + up_offsets, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    out_offsets = j * N + i_indices
    tl.store(aa_ptr + out_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process j sequentially, parallelize i dimension
    for j in range(1, N):
        num_elements = N - 1  # i from 1 to N-1
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, j, N, BLOCK_SIZE
        )