import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    j_indices = j_base + j_offsets
    
    for i in range(N):
        j_mask = (j_indices < i) & (j_indices < N)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            # Load aa[j, i] (transposed indices)
            aa_ji_ptrs = aa_ptr + j_indices * N + i
            aa_ji_vals = tl.load(aa_ji_ptrs, mask=j_mask, other=0.0)
            
            # Load bb[i, j]
            bb_ij_ptrs = bb_ptr + i * N + j_indices
            bb_ij_vals = tl.load(bb_ij_ptrs, mask=j_mask, other=0.0)
            
            # Compute aa[i, j] = aa[j, i] + bb[i, j]
            result = aa_ji_vals + bb_ij_vals
            
            # Store to aa[i, j]
            aa_ij_ptrs = aa_ptr + i * N + j_indices
            tl.store(aa_ij_ptrs, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)