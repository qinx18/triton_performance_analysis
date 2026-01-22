import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process elements from j = i to N-1 in blocks
    for j_start in range(i, N, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = (j_offsets < N) & (j_offsets >= i)
        
        # Load bb[j][i] values
        bb_indices = j_offsets * N + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Calculate k values for each j
        k_vals = k + (j_offsets - i) * (j_offsets + i + 1) // 2
        
        # Load current flat_2d_array values
        current_vals = tl.load(flat_2d_array_ptr + k_vals, mask=mask, other=0.0)
        
        # Add bb values
        new_vals = current_vals + bb_vals
        
        # Store back
        tl.store(flat_2d_array_ptr + k_vals, new_vals, mask=mask)

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, N,
        BLOCK_SIZE=BLOCK_SIZE
    )