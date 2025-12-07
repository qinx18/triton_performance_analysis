import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially due to wavefront dependencies
    # Each block processes a range of i values
    block_start_i = tl.program_id(0) * BLOCK_SIZE
    
    for i_offset in range(BLOCK_SIZE):
        i = block_start_i + i_offset
        # Use conditional instead of break
        if i < N:
            # For each i, process all valid j values (j < i)
            for j in range(i):
                # Read aa[j][i] and bb[i][j]
                aa_ji_offset = j * N + i
                bb_ij_offset = i * N + j
                aa_ij_offset = i * N + j
                
                aa_ji_val = tl.load(aa_ptr + aa_ji_offset)
                bb_ij_val = tl.load(bb_ptr + bb_ij_offset)
                
                # Compute aa[i][j] = aa[j][i] + bb[i][j]
                result = aa_ji_val + bb_ij_val
                tl.store(aa_ptr + aa_ij_offset, result)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 16
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    s114_kernel[(grid_size,)](
        aa, bb, N, BLOCK_SIZE
    )
    
    return aa