import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # Calculate linear indices for aa[i][j] and bb[i][j]
        aa_linear_idx = i * N + j_idx
        bb_linear_idx = i * N + j_idx
        
        # Calculate linear indices for cc[j][i] - note the transpose
        cc_linear_idx = j_idx * N + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_linear_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_linear_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_linear_idx, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result back to aa[i][j]
        tl.store(aa_ptr + aa_linear_idx, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE
    )