import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb, cc, flat_2d_array, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Process all j values in parallel using vectorized operations
        for j_block_start in range(1, LEN_2D, BLOCK_SIZE):
            # Calculate current j indices
            j_offsets = j_block_start + offsets
            j_mask = (j_offsets < LEN_2D) & (j_offsets >= 1)
            
            # Calculate k values for current j indices
            k_offsets = i * LEN_2D + j_offsets
            
            # Load bb[j-1][i] values
            bb_prev_offsets = (j_offsets - 1) * LEN_2D + i
            bb_prev_vals = tl.load(bb + bb_prev_offsets, mask=j_mask)
            
            # Load flat_2d_array[k-1] values
            flat_offsets = k_offsets - 1
            flat_vals = tl.load(flat_2d_array + flat_offsets, mask=j_mask)
            
            # Load cc[j][i] values
            cc_offsets = j_offsets * LEN_2D + i
            cc_vals = tl.load(cc + cc_offsets, mask=j_mask)
            
            # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
            result = bb_prev_vals + flat_vals * cc_vals
            
            # Store results
            bb_store_offsets = j_offsets * LEN_2D + i
            tl.store(bb + bb_store_offsets, result, mask=j_mask)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel
    grid = (1,)
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )