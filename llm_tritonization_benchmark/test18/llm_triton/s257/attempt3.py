import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] once
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # For the overwrite pattern, process j values sequentially
    # Each j iteration overwrites a[i], so we need the final value
    for j_local in range(BLOCK_SIZE):
        j_idx = pid * BLOCK_SIZE + j_local
        if j_idx < LEN_2D:
            # Load aa[j][i] where aa is row-major: aa[j][i] = aa[j * LEN_2D + i]
            aa_offset = j_idx * LEN_2D + i_val
            aa_ji = tl.load(aa_ptr + aa_offset)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_i = aa_ji - a_i_minus_1
            tl.store(a_ptr + i_val, a_i)
            
            # Load bb[j][i]
            bb_offset = j_idx * LEN_2D + i_val
            bb_ji = tl.load(bb_ptr + bb_offset)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            new_aa_ji = a_i + bb_ji
            tl.store(aa_ptr + aa_offset, new_aa_ji)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Sequential loop over i from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        s257_kernel[(grid_size,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )