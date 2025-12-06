import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential processing by diagonals to handle the dependency aa[i][j] = aa[i-1][j-1] + bb[i][j]
    
    # Process each diagonal sequentially
    for diag in range(2, 2 * LEN_2D - 1):  # diagonals from 2 to 2*LEN_2D-2
        # Calculate the range of valid (i, j) pairs for this diagonal where i+j = diag
        min_i = max(1, diag - (LEN_2D - 1))
        max_i = min(LEN_2D - 1, diag - 1)
        
        if min_i <= max_i:
            # Get block of work for this program
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            
            # Generate offsets for this block
            offsets = tl.arange(0, BLOCK_SIZE)
            i_indices = min_i + block_start + offsets
            
            # Mask for valid indices
            mask = (i_indices <= max_i) & (i_indices >= min_i)
            
            # Calculate corresponding j indices
            j_indices = diag - i_indices
            
            # Additional mask to ensure j is in valid range
            valid_mask = mask & (j_indices >= 1) & (j_indices < LEN_2D)
            
            # Load aa[i-1][j-1] values
            prev_i = i_indices - 1
            prev_j = j_indices - 1
            prev_offsets = prev_i * LEN_2D + prev_j
            aa_prev = tl.load(aa_ptr + prev_offsets, mask=valid_mask, other=0.0)
            
            # Load bb[i][j] values
            bb_offsets = i_indices * LEN_2D + j_indices
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
            
            # Compute new values
            new_vals = aa_prev + bb_vals
            
            # Store aa[i][j] values
            aa_offsets = i_indices * LEN_2D + j_indices
            tl.store(aa_ptr + aa_offsets, new_vals, mask=valid_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process each diagonal with sufficient parallelism
    for diag in range(2, 2 * LEN_2D - 1):
        min_i = max(1, diag - (LEN_2D - 1))
        max_i = min(LEN_2D - 1, diag - 1)
        
        if min_i <= max_i:
            num_elements = max_i - min_i + 1
            grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
            
            if grid_size > 0:
                s119_kernel[(grid_size,)](
                    aa, bb, LEN_2D, BLOCK_SIZE
                )
    
    return aa