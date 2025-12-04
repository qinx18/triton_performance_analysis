import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential computation - no parallelization due to dependencies
    for j in range(1, LEN_2D):
        block_start = tl.program_id(0) * BLOCK_SIZE
        i_offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
        
        # Process blocks sequentially within each row
        for block_idx in range(0, triton.cdiv(LEN_2D - 1, BLOCK_SIZE)):
            current_block_start = block_idx * BLOCK_SIZE + 1
            current_i_offsets = current_block_start + tl.arange(0, BLOCK_SIZE)
            mask = current_i_offsets < LEN_2D
            
            # Load aa[j][i-1]
            left_offsets = j * LEN_2D + (current_i_offsets - 1)
            left_mask = mask & (current_i_offsets > 0)
            left_vals = tl.load(aa_ptr + left_offsets, mask=left_mask, other=0.0)
            
            # Load aa[j-1][i]
            up_offsets = (j - 1) * LEN_2D + current_i_offsets
            up_vals = tl.load(aa_ptr + up_offsets, mask=mask, other=0.0)
            
            # Compute new values
            new_vals = (left_vals + up_vals) / 1.9
            
            # Store aa[j][i]
            store_offsets = j * LEN_2D + current_i_offsets
            tl.store(aa_ptr + store_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch single program since computation must be sequential
    grid = (1,)
    
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa