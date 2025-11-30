import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the current j position (column)
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Process elements from i=j to LEN_2D-1 for this column j
    i_start = j
    i_end = LEN_2D
    
    # Process in blocks along the i dimension
    for i_block_start in range(i_start, i_end, BLOCK_SIZE):
        # Calculate the range of i indices for this block
        i_offsets = tl.arange(0, BLOCK_SIZE) + i_block_start
        
        # Create mask to ensure we don't go beyond bounds
        mask = (i_offsets < i_end) & (i_offsets >= i_start)
        
        # Calculate memory offsets (row-major: [i][j] = i * LEN_2D + j)
        mem_offsets = i_offsets * LEN_2D + j
        
        # Load data with masking
        bb_vals = tl.load(bb_ptr + mem_offsets, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + mem_offsets, mask=mask, other=0.0)
        
        # Perform computation
        aa_vals = bb_vals + cc_vals
        
        # Store result with masking
        tl.store(aa_ptr + mem_offsets, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with one thread block per column (j dimension)
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa