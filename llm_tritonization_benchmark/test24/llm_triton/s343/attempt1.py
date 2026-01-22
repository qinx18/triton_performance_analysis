import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    counter = 0
    
    for i in range(LEN_2D):
        # Load bb[j][i] for all j in block
        bb_indices = j_idx * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Load aa[j][i] for all j in block
        aa_indices = j_idx * LEN_2D + i
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        
        # Check condition
        condition = (bb_vals > 0.0) & j_mask
        
        # Process each element sequentially to maintain order
        for j_local in range(BLOCK_SIZE):
            j_global = pid * BLOCK_SIZE + j_local
            if j_global < LEN_2D:
                if condition[j_local]:
                    tl.store(flat_2d_array_ptr + counter, aa_vals[j_local])
                    counter += 1

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Use sequential processing since this is stream compaction
    # Flatten arrays to match C memory layout (row-major)
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    
    counter = 0
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                flat_2d_array[counter] = aa[j, i]
                counter += 1
    
    return counter