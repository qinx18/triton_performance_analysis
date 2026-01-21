import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_I: tl.constexpr):
    pid_i = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE_I columns
    i_start = pid_i * BLOCK_SIZE_I
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i_offsets < LEN_2D
    
    # Initialize k for each column i
    k_base = i_offsets * LEN_2D + 1
    
    for j in range(1, LEN_2D):
        # Calculate current k values for each column
        k_vals = k_base + (j - 1)
        
        # Load bb[j-1][i] for all columns
        bb_prev_offsets = (j - 1) * LEN_2D + i_offsets
        bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=i_mask, other=0.0)
        
        # Load flat_2d_array[k-1] for each column
        flat_offsets = k_vals - 1
        flat_vals = tl.load(flat_2d_array_ptr + flat_offsets, mask=i_mask, other=0.0)
        
        # Load cc[j][i] for all columns
        cc_offsets = j * LEN_2D + i_offsets
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_vals * cc_vals
        
        # Store bb[j][i]
        bb_offsets = j * LEN_2D + i_offsets
        tl.store(bb_ptr + bb_offsets, result, mask=i_mask)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE_I = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )