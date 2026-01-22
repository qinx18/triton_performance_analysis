import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_start = pid * BLOCK_SIZE
    i_idx = i_start + i_offsets
    
    for j in range(N):
        # Only process elements where i >= j
        mask = (i_idx < N) & (i_idx >= j)
        
        if tl.sum(mask) > 0:
            # Calculate linear indices for row-major 2D arrays
            linear_idx = i_idx * N + j
            
            # Load bb and cc values
            bb_vals = tl.load(bb_ptr + linear_idx, mask=mask, other=0.0)
            cc_vals = tl.load(cc_ptr + linear_idx, mask=mask, other=0.0)
            
            # Compute aa[i][j] = bb[i][j] + cc[i][j]
            result = bb_vals + cc_vals
            
            # Store result
            tl.store(aa_ptr + linear_idx, result, mask=mask)

def s1232_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        N, BLOCK_SIZE=BLOCK_SIZE
    )