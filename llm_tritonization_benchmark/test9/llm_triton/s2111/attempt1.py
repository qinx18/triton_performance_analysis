import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential wavefront computation - must be done sequentially
    for j in range(1, LEN_2D):
        # Process row j in blocks
        for block_start in range(1, LEN_2D, BLOCK_SIZE):
            # Get thread block info
            pid = tl.program_id(0)
            block_offset = pid * BLOCK_SIZE
            
            # Only process if this block should handle this iteration
            if block_offset == (block_start - 1) // BLOCK_SIZE * BLOCK_SIZE:
                offsets = tl.arange(0, BLOCK_SIZE)
                i_offsets = block_start + offsets
                
                # Mask for valid indices
                mask = (i_offsets < LEN_2D) & (i_offsets >= 1)
                
                # Calculate memory offsets for aa[j][i], aa[j][i-1], aa[j-1][i]
                current_offsets = j * LEN_2D + i_offsets
                left_offsets = j * LEN_2D + (i_offsets - 1)
                up_offsets = (j - 1) * LEN_2D + i_offsets
                
                # Load values
                left_vals = tl.load(aa_ptr + left_offsets, mask=mask)
                up_vals = tl.load(aa_ptr + up_offsets, mask=mask)
                
                # Compute new values
                new_vals = (left_vals + up_vals) / 1.9
                
                # Store results
                tl.store(aa_ptr + current_offsets, new_vals, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel with single program since computation must be sequential
    grid = (1,)
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1
    )
    
    return aa