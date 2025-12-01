import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, flat_2d_array_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(LEN_2D):
        # For this i value, process all j values in parallel
        for block_start in range(1, LEN_2D, BLOCK_SIZE):
            j_offsets = block_start + offsets
            j_mask = (j_offsets < LEN_2D) & (j_offsets >= 1)
            
            # Calculate k values for each j
            k_vals = i * LEN_2D + j_offsets
            
            # Load bb[j-1][i] values
            bb_prev_offsets = (j_offsets - 1) * LEN_2D + i
            bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=j_mask, other=0.0)
            
            # Load flat_2d_array[k-1] values
            flat_offsets = k_vals - 1
            flat_vals = tl.load(flat_2d_array_ptr + flat_offsets, mask=j_mask, other=0.0)
            
            # Load cc[j][i] values
            cc_offsets = j_offsets * LEN_2D + i
            cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
            
            # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
            result = bb_prev_vals + flat_vals * cc_vals
            
            # Store bb[j][i] values
            bb_curr_offsets = j_offsets * LEN_2D + i
            tl.store(bb_ptr + bb_curr_offsets, result, mask=j_mask)

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel
    s126_kernel[(1,)](
        bb, flat_2d_array, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )