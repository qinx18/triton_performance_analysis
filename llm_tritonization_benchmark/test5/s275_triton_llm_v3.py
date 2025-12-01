import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Check condition aa[0][i] > 0
    condition_val = tl.load(aa_ptr + i)
    
    if condition_val > 0.0:
        # Sequential computation for j = 1 to LEN_2D-1
        for j in range(1, LEN_2D):
            # Load aa[j-1][i]
            prev_val = tl.load(aa_ptr + (j-1) * LEN_2D + i)
            
            # Load bb[j][i] and cc[j][i]
            bb_val = tl.load(bb_ptr + j * LEN_2D + i)
            cc_val = tl.load(cc_ptr + j * LEN_2D + i)
            
            # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            new_val = prev_val + bb_val * cc_val
            
            # Store aa[j][i]
            tl.store(aa_ptr + j * LEN_2D + i, new_val)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )