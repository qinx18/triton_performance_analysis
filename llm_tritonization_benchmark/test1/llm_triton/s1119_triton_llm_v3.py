import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    i_val,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Calculate offsets for 2D arrays
    write_offsets = i_val * LEN_2D + j_offsets
    read_offsets = (i_val - 1) * LEN_2D + j_offsets
    
    # Load values from read-only copy and bb array
    aa_prev = tl.load(aa_copy_ptr + read_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + write_offsets, mask=mask)
    
    # Compute result
    result = aa_prev + bb_vals
    
    # Store result to original aa array
    tl.store(aa_ptr + write_offsets, result, mask=mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy of aa array
    aa_copy = aa.clone()
    
    # Sequential loop over i dimension
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](
            aa,
            aa_copy,
            bb,
            i_val,
            LEN_2D,
            BLOCK_SIZE,
        )