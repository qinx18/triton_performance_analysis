import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE_I: tl.constexpr):
    # Each program handles a block of i values
    block_start_i = tl.program_id(0) * BLOCK_SIZE_I
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    i_vals = block_start_i + i_offsets
    i_mask = i_vals < LEN_2D
    
    # Process each i in this block
    for i_idx in range(BLOCK_SIZE_I):
        i = block_start_i + i_idx
        if i >= LEN_2D:
            return
            
        # Calculate starting k for this i
        k = (i + 1) * i // 2 + i
        
        # Inner j loop - process sequentially since k depends on previous j
        for j in range(i, LEN_2D):
            # Load bb[j][i]
            bb_offset = j * LEN_2D + i
            bb_val = tl.load(bb_ptr + bb_offset)
            
            # Load current flat_2d_array[k], add bb_val, store back
            current_val = tl.load(flat_2d_array_ptr + k)
            new_val = current_val + bb_val
            tl.store(flat_2d_array_ptr + k, new_val)
            
            # Update k for next iteration
            k += j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE_I = 32
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s141_kernel[grid](
        bb, flat_2d_array, 
        LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )