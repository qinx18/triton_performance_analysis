import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        # Calculate memory offsets
        curr_offsets = i * LEN_2D + j_idx
        prev_offsets = (i - 1) * LEN_2D + j_idx
        
        # Load data
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + curr_offsets, mask=j_mask, other=0.0)
        
        # Compute and store
        result = aa_prev + bb_curr
        tl.store(aa_ptr + curr_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa