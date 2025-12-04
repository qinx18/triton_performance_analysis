import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential computation by diagonals to handle dependencies
    for diag in range(2, 2 * LEN_2D - 1):
        # Calculate how many elements are in this diagonal
        diag_size = min(diag - 1, 2 * LEN_2D - diag - 1, LEN_2D - 1)
        
        # Process diagonal in blocks
        num_blocks = triton.cdiv(diag_size, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            block_offsets = tl.arange(0, BLOCK_SIZE)
            elem_idx = block_idx * BLOCK_SIZE + block_offsets
            mask = elem_idx < diag_size
            
            # Calculate j and i for this diagonal position
            # For diagonal d, we have j + i = d
            # Starting from j=1, so j ranges from max(1, d-LEN_2D+1) to min(d-1, LEN_2D-1)
            j_start = max(1, diag - LEN_2D + 1)
            j_vals = j_start + elem_idx
            i_vals = diag - j_vals
            
            # Check bounds
            valid_mask = mask & (j_vals < LEN_2D) & (i_vals >= 1) & (i_vals < LEN_2D) & (j_vals >= 1)
            
            # Load aa[j][i-1]
            left_offsets = j_vals * LEN_2D + (i_vals - 1)
            left_vals = tl.load(aa_ptr + left_offsets, mask=valid_mask, other=0.0)
            
            # Load aa[j-1][i]
            up_offsets = (j_vals - 1) * LEN_2D + i_vals
            up_vals = tl.load(aa_ptr + up_offsets, mask=valid_mask, other=0.0)
            
            # Compute new values
            new_vals = (left_vals + up_vals) / 1.9
            
            # Store aa[j][i]
            store_offsets = j_vals * LEN_2D + i_vals
            tl.store(aa_ptr + store_offsets, new_vals, mask=valid_mask)

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