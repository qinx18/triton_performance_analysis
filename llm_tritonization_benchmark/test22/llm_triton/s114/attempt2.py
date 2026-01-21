import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_start = pid * BLOCK_SIZE
    j_indices = j_start + j_offsets
    
    for i in range(n):
        j_mask = (j_indices < i) & (j_indices >= 0) & (j_indices < n)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Read aa[j][i] and bb[i][j]
            aa_ji_ptrs = aa_ptr + j_indices * n + i
            bb_ij_ptrs = bb_ptr + i * n + j_indices
            
            aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
            bb_ij_vals = tl.load(bb_ij_ptrs, mask=j_mask, other=0.0)
            
            # Compute aa[i][j] = aa[j][i] + bb[i][j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store to aa[i][j]
            aa_ij_ptrs = aa_ptr + i * n + j_indices
            tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    n = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, n, BLOCK_SIZE=BLOCK_SIZE)