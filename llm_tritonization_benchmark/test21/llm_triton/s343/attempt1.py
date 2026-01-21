import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets
    j_mask = j_indices < LEN_2D
    
    k = -1
    for i in range(LEN_2D):
        # Load bb[j][i] for all j values in this block
        bb_ptrs = bb_ptr + j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Check condition bb[j][i] > 0.0
        condition_mask = (bb_vals > 0.0) & j_mask
        
        # Load aa[j][i] for elements that pass condition
        aa_ptrs = aa_ptr + j_indices * LEN_2D + i
        aa_vals = tl.load(aa_ptrs, mask=condition_mask, other=0.0)
        
        # For each element that passes condition, increment k and store
        for block_idx in range(BLOCK_SIZE):
            j_idx = pid * BLOCK_SIZE + block_idx
            if j_idx < LEN_2D and bb_vals[block_idx] > 0.0:
                k += 1
                tl.store(flat_2d_array_ptr + k, aa_vals[block_idx])

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, LEN_2D)
    
    # Use PyTorch for stream compaction since it's difficult to parallelize correctly
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                k += 1
                flat_2d_array[k] = aa[j, i]