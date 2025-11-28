import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(
    aa_ptr, aa_copy_ptr, bb_ptr, bb_copy_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get column index (i dimension)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process j dimension sequentially for each column i
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_offset = j * LEN_2D + i
        aa_prev_offset = (j - 1) * LEN_2D + i
        cc_offset = j * LEN_2D + i
        
        aa_prev_val = tl.load(aa_copy_ptr + aa_prev_offset)
        cc_val = tl.load(cc_ptr + cc_offset)
        aa_result = aa_prev_val + cc_val
        tl.store(aa_ptr + aa_offset, aa_result)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_offset = j * LEN_2D + i
        bb_prev_offset = j * LEN_2D + (i - 1)
        
        bb_prev_val = tl.load(bb_copy_ptr + bb_prev_offset)
        bb_result = bb_prev_val + cc_val
        tl.store(bb_ptr + bb_offset, bb_result)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copies to handle WAR dependencies
    aa_copy = aa.clone()
    bb_copy = bb.clone()
    
    # Flatten 2D arrays for kernel access
    aa_flat = aa.view(-1)
    aa_copy_flat = aa_copy.view(-1)
    bb_flat = bb.view(-1)
    bb_copy_flat = bb_copy.view(-1)
    cc_flat = cc.view(-1)
    
    # Launch kernel with one thread per column (i dimension)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 256
    
    s233_kernel[grid](
        aa_flat, aa_copy_flat, bb_flat, bb_copy_flat, cc_flat,
        LEN_2D, BLOCK_SIZE
    )