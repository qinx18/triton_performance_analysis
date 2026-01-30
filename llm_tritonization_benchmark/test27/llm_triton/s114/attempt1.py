import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    
    for i in range(len_2d):
        j_indices = j_base + j_offsets
        mask = (j_indices < i) & (j_indices >= 0)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            # Calculate memory addresses
            aa_ij_addrs = aa_ptr + i * len_2d + j_indices
            aa_ji_addrs = aa_ptr + j_indices * len_2d + i
            bb_ij_addrs = bb_ptr + i * len_2d + j_indices
            
            # Load values
            aa_ji_vals = tl.load(aa_ji_addrs, mask=mask)
            bb_ij_vals = tl.load(bb_ij_addrs, mask=mask)
            
            # Compute and store
            result = aa_ji_vals + bb_ij_vals
            tl.store(aa_ij_addrs, result, mask=mask)

def s114_triton(aa, bb, len_2d):
    BLOCK_SIZE = 256
    max_j = len_2d - 1
    grid = (triton.cdiv(max_j, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )