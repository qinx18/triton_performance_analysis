import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= n:
        return
    
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = i * (i + 1) / 2
    k_start = i * (i + 1) // 2
    
    # Process j from i to n-1
    num_j = n - i
    
    # Process in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(0, num_j, BLOCK_SIZE):
        current_j = i + j_block + j_offsets
        mask = (j_block + j_offsets) < num_j
        
        # Calculate k for each j
        # k starts at k_start and increases by j+1 at each step
        # For j=i: k = k_start
        # For j=i+1: k = k_start + (i+1)
        # For j=i+2: k = k_start + (i+1) + (i+2)
        # General: k = k_start + sum from j=i to current_j of j
        j_offset = j_block + j_offsets
        k_increment = j_offset * (2 * i + j_offset + 1) // 2
        k_indices = k_start + k_increment
        
        # Load bb[j][i] values
        bb_indices = current_j * n + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=mask, other=0.0)
        
        # Update
        new_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_indices, new_vals, mask=mask)

def s141_triton(bb, flat_2d_array):
    n = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (n,)
    
    s141_kernel[grid](
        flat_2d_array, bb, n, BLOCK_SIZE
    )