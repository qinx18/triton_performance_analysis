import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1 = i * (i+1) / 2 + i
    k = i * (i + 1) // 2 + i
    
    # Process elements from j=i to LEN_2D-1 in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = (current_j < LEN_2D) & (current_j >= i)
        
        # Calculate k values for this block
        # k starts at initial value for j=i, then increments by j+1 for each step
        k_increments = tl.arange(0, BLOCK_SIZE)
        k_base = k
        
        # Calculate cumulative increments for k
        for offset in range(BLOCK_SIZE):
            j_val = j_start + offset
            if j_val >= i and j_val < LEN_2D:
                if offset > 0:
                    k_base += j_val
        
        k_values = k + k_increments * 0  # Initialize with base k
        
        # Calculate proper k values by adding increments
        for offset in range(BLOCK_SIZE):
            j_val = j_start + offset
            if j_val > i and j_val < LEN_2D:
                increment = 0
                for prev_j in range(i, j_val):
                    increment += prev_j + 1
                k_values = tl.where(k_increments == offset, k + increment, k_values)
        
        # Load bb[j][i] values
        bb_indices = current_j * LEN_2D + i
        bb_mask = j_mask & (bb_indices < LEN_2D * LEN_2D)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=bb_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_mask = j_mask & (k_values < LEN_2D * LEN_2D) & (k_values >= 0)
        current_vals = tl.load(flat_2d_array_ptr + k_values, mask=flat_mask, other=0.0)
        
        # Update flat_2d_array
        new_vals = current_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_values, new_vals, mask=flat_mask)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 32
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )