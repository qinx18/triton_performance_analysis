import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get column index (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Check condition: aa[0][i] > 0
    condition_offset = i  # aa[0][i]
    condition_val = tl.load(aa_copy_ptr + condition_offset)
    
    if condition_val > 0.0:
        # Process j dimension sequentially from 1 to LEN_2D-1
        for j in range(1, LEN_2D):
            # Calculate offsets for aa[j][i], aa[j-1][i], bb[j][i], cc[j][i]
            curr_offset = j * LEN_2D + i      # aa[j][i]
            prev_offset = (j-1) * LEN_2D + i  # aa[j-1][i]
            bb_offset = j * LEN_2D + i        # bb[j][i]
            cc_offset = j * LEN_2D + i        # cc[j][i]
            
            # Load values - use copy for reading aa
            prev_val = tl.load(aa_copy_ptr + prev_offset)
            bb_val = tl.load(bb_ptr + bb_offset)
            cc_val = tl.load(cc_ptr + cc_offset)
            
            # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            result = prev_val + bb_val * cc_val
            
            # Store to original array
            tl.store(aa_ptr + curr_offset, result)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy for WAR dependency handling
    aa_copy = aa.clone()
    
    # Launch kernel with one thread per column (i dimension)
    grid = (LEN_2D,)
    BLOCK_SIZE = 256
    
    s275_kernel[grid](
        aa,
        aa_copy,
        bb,
        cc,
        LEN_2D,
        BLOCK_SIZE,
    )