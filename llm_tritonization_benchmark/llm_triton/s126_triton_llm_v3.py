import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, flat_2d_array_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    i_block_id = tl.program_id(0)
    
    i_start = i_block_id * BLOCK_SIZE_I
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i_offsets < LEN_2D
    
    for i_idx in range(BLOCK_SIZE_I):
        i = i_start + i_idx
        if i >= LEN_2D:
            break
            
        k = i * LEN_2D + 1
        
        for j in range(1, LEN_2D, BLOCK_SIZE_J):
            j_offsets = j + tl.arange(0, BLOCK_SIZE_J)
            j_mask = j_offsets < LEN_2D
            
            # Load bb[j-1][i] values
            bb_prev_offsets = (j_offsets - 1) * LEN_2D + i
            bb_prev_mask = j_mask & (j_offsets > 0)
            bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=bb_prev_mask, other=0.0)
            
            # Load flat_2d_array[k-1] values
            flat_offsets = k + tl.arange(0, BLOCK_SIZE_J) - 1
            flat_mask = j_mask & (flat_offsets >= 0) & (flat_offsets < LEN_2D * LEN_2D)
            flat_vals = tl.load(flat_2d_array_ptr + flat_offsets, mask=flat_mask, other=0.0)
            
            # Load cc[j][i] values
            cc_offsets = j_offsets * LEN_2D + i
            cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
            
            # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
            bb_new = bb_prev + flat_vals * cc_vals
            
            # Store bb[j][i] values
            bb_store_offsets = j_offsets * LEN_2D + i
            tl.store(bb_ptr + bb_store_offsets, bb_new, mask=j_mask)
            
            k += BLOCK_SIZE_J

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE_I = 1  # Sequential in i due to k dependency
    BLOCK_SIZE_J = 64  # Parallel in j blocks
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s126_kernel[grid](
        bb, flat_2d_array, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )
    
    return bb