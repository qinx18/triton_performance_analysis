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
    # Get column index
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i < LEN_2D
    
    # Check condition: aa[0][i] > 0
    aa_0_offset = i
    aa_0_vals = tl.load(aa_copy_ptr + aa_0_offset, mask=i_mask, other=0.0)
    condition = aa_0_vals > 0.0
    
    # Process each column that meets the condition
    for col_idx in range(BLOCK_SIZE):
        if col_idx < LEN_2D:
            actual_i = tl.program_id(0) * BLOCK_SIZE + col_idx
            if actual_i < LEN_2D and condition[col_idx]:
                # Sequential loop over j (cannot be parallelized due to dependency)
                for j in range(1, LEN_2D):
                    # aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
                    prev_offset = (j - 1) * LEN_2D + actual_i
                    curr_offset = j * LEN_2D + actual_i
                    
                    # Read aa[j-1][i] from copy (for original values)
                    prev_val = tl.load(aa_copy_ptr + prev_offset)
                    
                    # Read bb[j][i] and cc[j][i]
                    bb_val = tl.load(bb_ptr + curr_offset)
                    cc_val = tl.load(cc_ptr + curr_offset)
                    
                    # Compute new value
                    new_val = prev_val + bb_val * cc_val
                    
                    # Store to original array
                    tl.store(aa_ptr + curr_offset, new_val)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of aa to handle WAR dependencies
    aa_copy = aa.clone()
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa,
        aa_copy,
        bb,
        cc,
        LEN_2D,
        BLOCK_SIZE,
    )