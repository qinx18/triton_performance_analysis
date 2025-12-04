import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation - process row by row, then within each row
    for j in range(1, LEN_2D):
        # Process each row in blocks
        num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE + 1
            i_offsets = block_start + offsets
            mask = i_offsets < LEN_2D
            
            # Load aa[j][i-1]
            left_offsets = j * LEN_2D + (i_offsets - 1)
            left_mask = mask & (i_offsets >= 1)
            left_vals = tl.load(aa_ptr + left_offsets, mask=left_mask, other=0.0)
            
            # Load aa[j-1][i]
            up_offsets = (j - 1) * LEN_2D + i_offsets
            up_vals = tl.load(aa_ptr + up_offsets, mask=mask, other=0.0)
            
            # Compute new values
            new_vals = (left_vals + up_vals) / 1.9
            
            # Store aa[j][i]
            store_offsets = j * LEN_2D + i_offsets
            tl.store(aa_ptr + store_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single program since computation must be sequential
    grid = (1,)
    
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa