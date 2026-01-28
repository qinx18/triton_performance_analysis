import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which i this block handles
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Calculate starting k for this i
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = i * (i + 1) // 2 + i
    
    # Process j loop from i to LEN_2D-1 in blocks
    j_range = LEN_2D - i
    
    # Pre-compute offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, j_range, BLOCK_SIZE):
        # Current j values for this block
        j_offsets = i + block_start + offsets
        j_mask = j_offsets < LEN_2D
        
        # Calculate k values for this block
        # k increases by j+1 for each step, so k[n] = k_start + sum(j+1) for j in [i, i+n)
        # For j from i to i+block_start+offset: k = k_start + sum from i to current_j of (j+1)
        # sum(j+1) from i to current_j = sum(j) from i to current_j + count
        # sum(j) from i to current_j = (current_j*(current_j+1) - (i-1)*i)/2
        current_k_offsets = k_start + ((j_offsets * (j_offsets + 1) - i * (i - 1)) // 2) + block_start + offsets
        
        # Load bb values: bb[j][i]
        bb_offsets = j_offsets * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + current_k_offsets, mask=j_mask, other=0.0)
        
        # Add bb values
        new_vals = flat_vals + bb_vals
        
        # Store back
        tl.store(flat_2d_array_ptr + current_k_offsets, new_vals, mask=j_mask)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread block per i
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array, 
        LEN_2D, BLOCK_SIZE
    )