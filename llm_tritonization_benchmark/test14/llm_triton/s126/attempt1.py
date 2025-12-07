import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_I: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(axis=0)
    
    # Calculate i index
    i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i < LEN_2D
    
    # Sequential loop over j (cannot parallelize due to dependency)
    for j in range(1, LEN_2D):
        # Calculate k based on i and j
        k = i * LEN_2D + j
        
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset, mask=i_mask, other=0.0)
        
        # Load flat_2d_array[k-1]
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset, mask=i_mask, other=0.0)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset, mask=i_mask, other=0.0)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_curr_offset = j * LEN_2D + i
        tl.store(bb_ptr + bb_curr_offset, result, mask=i_mask)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch configuration
    BLOCK_SIZE_I = 32
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    # Launch kernel
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )