import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Calculate starting k value for this i
    k = (i + 1) * i // 2 + i
    
    # Process elements for this i in blocks
    for j_start in range(i, N, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets < N) & (j_offsets >= i)
        
        # Calculate k values for this block
        k_offsets = k + j_offsets * (j_offsets + 1) // 2 - i * (i + 1) // 2
        
        # Load bb values
        bb_offsets = j_offsets * N + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_offsets, mask=j_mask, other=0.0)
        
        # Update flat_2d_array
        result = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_offsets, result, mask=j_mask)
        
        # Update k for next iteration
        k += tl.sum(tl.where(j_mask, j_offsets + 1, 0))

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, 
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )