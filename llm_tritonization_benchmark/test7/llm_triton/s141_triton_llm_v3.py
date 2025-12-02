import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Calculate starting k for this i
    k = (i + 1) * (i + 1 - 1) // 2 + i
    
    # Process inner loop in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block_start in range(i, LEN_2D, BLOCK_SIZE):
        j_indices = j_block_start + j_offsets
        j_mask = (j_indices < LEN_2D) & (j_indices >= i)
        
        if tl.sum(j_mask.to(tl.int32)) == 0:
            continue
            
        # Load bb[j][i] values
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Calculate k values for this block
        k_vals = k + tl.cumsum(j_indices + 1, axis=0) - (i + 1)
        
        # Update flat_2d_array[k] values
        for idx in range(BLOCK_SIZE):
            if j_block_start + idx < LEN_2D and j_block_start + idx >= i:
                j = j_block_start + idx
                k_val = k + tl.sum(tl.arange(0, j - i + 1) + i + 1)
                old_val = tl.load(flat_2d_array_ptr + k_val)
                bb_val = tl.load(bb_ptr + j * LEN_2D + i)
                tl.store(flat_2d_array_ptr + k_val, old_val + bb_val)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 32
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )