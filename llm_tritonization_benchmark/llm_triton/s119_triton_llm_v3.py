import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    i_val,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # Calculate 2D array offsets
    read_offsets = (i_val - 1) * LEN_2D + (j_offsets - 1)
    write_offsets = i_val * LEN_2D + j_offsets
    bb_offsets = i_val * LEN_2D + j_offsets
    
    # Load from read-only copy and bb array
    aa_vals = tl.load(aa_copy_ptr + read_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute result
    result = aa_vals + bb_vals
    
    # Store to original array
    tl.store(aa_ptr + write_offsets, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy
    aa_copy = aa.clone()
    
    # Sequential loop over i, parallel over j
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s119_kernel[grid](
            aa,
            aa_copy,
            bb,
            i_val,
            LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE,
        )