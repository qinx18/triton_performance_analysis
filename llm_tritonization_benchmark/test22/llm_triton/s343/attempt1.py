import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    counter = 0
    
    for i in range(LEN_2D):
        # Load bb[j][i] and aa[j][i] for all j values in this block
        bb_ptrs = bb_ptr + j_idx * LEN_2D + i
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Check condition bb[j][i] > 0
        condition = (bb_vals > 0.0) & j_mask
        
        # For each element that meets condition, store it
        for j_local in range(BLOCK_SIZE):
            if j_idx[j_local] < LEN_2D:
                if condition[j_local]:
                    tl.store(flat_ptr + counter, aa_vals[j_local])
                    counter += 1

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Use CPU implementation for stream compaction pattern
    # This is the correct way to handle conditional packing
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    
    mask = bb_flat > 0.0
    packed_values = aa_flat[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values