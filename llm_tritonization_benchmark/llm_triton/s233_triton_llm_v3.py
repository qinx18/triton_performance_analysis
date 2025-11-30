import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current column (i)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process rows in blocks for the first loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        block_size = j_end - j_start
        
        if block_size <= 0:
            break
            
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        
        # Calculate offsets for current and previous rows
        curr_offsets = j_offsets * LEN_2D + i
        prev_offsets = (j_offsets - 1) * LEN_2D + i
        
        # Load from read-only copies
        aa_prev_vals = tl.load(aa_copy_ptr + prev_offsets, mask=mask)
        cc_vals = tl.load(cc_ptr + curr_offsets, mask=mask)
        
        # Compute and store to original array
        aa_result = aa_prev_vals + cc_vals
        tl.store(aa_ptr + curr_offsets, aa_result, mask=mask)
    
    # Process rows in blocks for the second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        block_size = j_end - j_start
        
        if block_size <= 0:
            break
            
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        
        # Calculate offsets for current and previous columns
        curr_offsets = j_offsets * LEN_2D + i
        prev_offsets = j_offsets * LEN_2D + (i - 1)
        
        # Load from read-only copies
        bb_prev_vals = tl.load(bb_copy_ptr + prev_offsets, mask=mask)
        cc_vals = tl.load(cc_ptr + curr_offsets, mask=mask)
        
        # Compute and store to original array
        bb_result = bb_prev_vals + cc_vals
        tl.store(bb_ptr + curr_offsets, bb_result, mask=mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    # Launch kernel with one thread per column (excluding first column)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 32
    
    s233_kernel[grid](
        aa, aa_copy, bb, bb_copy, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )