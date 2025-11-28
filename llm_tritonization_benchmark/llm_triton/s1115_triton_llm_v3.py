import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, 
                 BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offset = pid_i * BLOCK_SIZE_I
    j_offset = pid_j * BLOCK_SIZE_J
    
    i_range = tl.arange(0, BLOCK_SIZE_I)
    j_range = tl.arange(0, BLOCK_SIZE_J)
    
    i_indices = i_offset + i_range
    j_indices = j_offset + j_range
    
    i_mask = i_indices < LEN_2D
    j_mask = j_indices < LEN_2D
    
    i_indices = tl.expand_dims(i_indices, 1)
    j_indices = tl.expand_dims(j_indices, 0)
    
    aa_indices = i_indices * LEN_2D + j_indices
    bb_indices = i_indices * LEN_2D + j_indices
    cc_indices = j_indices * LEN_2D + i_indices
    
    mask = tl.expand_dims(i_mask, 1) & tl.expand_dims(j_mask, 0)
    
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
    
    result = aa_vals * cc_vals + bb_vals
    
    tl.store(aa_ptr + aa_indices, result, mask=mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE_I = min(16, LEN_2D)
    BLOCK_SIZE_J = min(16, LEN_2D)
    
    grid_i = (LEN_2D + BLOCK_SIZE_I - 1) // BLOCK_SIZE_I
    grid_j = (LEN_2D + BLOCK_SIZE_J - 1) // BLOCK_SIZE_J
    grid = (grid_i, grid_j)
    
    s1115_kernel[grid](
        aa, bb, cc, LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )