import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_i = i < LEN_2D
    
    # Check condition aa[0][i] > 0
    aa_0_i = tl.load(aa_ptr + i, mask=mask_i)
    condition = aa_0_i > 0.0
    
    # Process each column where condition is true
    for idx in range(BLOCK_SIZE):
        actual_i = tl.program_id(0) * BLOCK_SIZE + idx
        if actual_i < LEN_2D and condition[idx]:
            # Sequential loop over j dimension (cannot be parallelized due to dependency)
            for j in range(1, LEN_2D):
                # Load aa[j-1][i]
                prev_val = tl.load(aa_ptr + (j-1) * LEN_2D + actual_i)
                # Load bb[j][i] and cc[j][i]
                bb_val = tl.load(bb_ptr + j * LEN_2D + actual_i)
                cc_val = tl.load(cc_ptr + j * LEN_2D + actual_i)
                # Compute and store aa[j][i]
                new_val = prev_val + bb_val * cc_val
                tl.store(aa_ptr + j * LEN_2D + actual_i, new_val)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, LEN_2D, BLOCK_SIZE
    )