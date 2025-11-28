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
    # Get the column index (i)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Check condition: aa[0][i] > 0
    aa_0_i_offset = 0 * LEN_2D + i
    aa_0_i = tl.load(aa_copy_ptr + aa_0_i_offset)
    
    if aa_0_i <= 0.0:
        return
    
    # Process column i in blocks
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        block_size = j_end - j_start
        
        if block_size <= 0:
            break
            
        # Create offsets for current block
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        mask = mask & (j_offsets >= 1)
        
        # Calculate memory offsets for aa[j][i], aa[j-1][i], bb[j][i], cc[j][i]
        aa_j_i_offsets = j_offsets * LEN_2D + i
        aa_j_minus_1_i_offsets = (j_offsets - 1) * LEN_2D + i
        bb_j_i_offsets = j_offsets * LEN_2D + i
        cc_j_i_offsets = j_offsets * LEN_2D + i
        
        # Load values
        aa_j_minus_1_i = tl.load(aa_copy_ptr + aa_j_minus_1_i_offsets, mask=mask)
        bb_j_i = tl.load(bb_ptr + bb_j_i_offsets, mask=mask)
        cc_j_i = tl.load(cc_ptr + cc_j_i_offsets, mask=mask)
        
        # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_j_minus_1_i + bb_j_i * cc_j_i
        
        # Store result
        tl.store(aa_ptr + aa_j_i_offsets, result, mask=mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Launch kernel with one thread per column
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s275_kernel[grid](
        aa,
        aa_copy,
        bb,
        cc,
        LEN_2D,
        BLOCK_SIZE,
    )