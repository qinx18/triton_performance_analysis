import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Calculate diagonal indices (i*N + i for aa[i][i])
    diag_indices = offsets * N + offsets
    
    # Load diagonal elements
    bb_vals = tl.load(bb_ptr + diag_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + diag_indices, mask=mask)
    aa_vals = tl.load(aa_ptr + diag_indices, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store back to aa
    tl.store(aa_ptr + diag_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2101_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)