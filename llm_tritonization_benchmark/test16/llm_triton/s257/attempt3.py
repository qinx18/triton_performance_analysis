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
    
    # Load a[i-1] as scalar
    a_i_minus_1 = tl.load(a_ptr + i_val - 1)
    
    # Load aa[j][i] values
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=0.0)
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_ji = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Compute a[i] = aa[j][i] - a[i-1] (last j iteration wins)
    new_a_i = aa_ji - a_i_minus_1
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    new_aa_ji = new_a_i + bb_ji
    
    # Store new aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_ji, mask=j_mask)
    
    # Store new a[i] values where mask is true
    # Create scalar mask for each valid j
    for block_offset in range(BLOCK_SIZE):
        if pid * BLOCK_SIZE + block_offset < LEN_2D:
            scalar_new_a_i = tl.load(aa_ptr + (pid * BLOCK_SIZE + block_offset) * LEN_2D + i_val) - a_i_minus_1
            tl.store(a_ptr + i_val, scalar_new_a_i)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to LEN_2D-1
    for i in range(1, LEN_2D):
        # Calculate grid size for j dimension
        grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
        
        # Launch kernel for all j values at this i
        s257_kernel[(grid_size,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )