import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    for j in range(N):
        # Only process elements where i >= j (triangular access pattern)
        valid_mask = (i_offsets >= j) & (i_offsets < N)
        
        # Calculate linear indices for aa[i][j], bb[i][j], cc[i][j]
        linear_idx = i_offsets * N + j
        
        # Load bb[i][j] and cc[i][j]
        bb_vals = tl.load(bb_ptr + linear_idx, mask=valid_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=valid_mask, other=0.0)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, aa_vals, mask=valid_mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1232_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)