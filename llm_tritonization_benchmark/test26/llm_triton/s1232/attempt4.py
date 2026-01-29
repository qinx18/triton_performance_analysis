import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(LEN_2D):
        i_idx = j + pid * BLOCK_SIZE + i_offsets
        
        # Only process elements where i < LEN_2D (i >= j is already satisfied by i_idx = j + ...)
        valid_mask = i_idx < LEN_2D
        
        # Compute linear indices for row-major layout: aa[i][j] = aa_ptr[i * LEN_2D + j]
        linear_idx = i_idx * LEN_2D + j
        
        # Load values with mask
        bb_vals = tl.load(bb_ptr + linear_idx, mask=valid_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=valid_mask, other=0.0)
        
        # Compute result
        result = bb_vals + cc_vals
        
        # Store result with mask
        tl.store(aa_ptr + linear_idx, result, mask=valid_mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Maximum i range for any j is j to LEN_2D-1
    # So maximum range size is LEN_2D (when j=0)
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )