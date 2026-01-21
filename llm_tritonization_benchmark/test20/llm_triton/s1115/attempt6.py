import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_j = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid_j * BLOCK_SIZE + j_offsets
    j_mask = j_indices < N
    
    for i in range(N):
        aa_base = i * N
        bb_base = i * N
        
        # Load aa[i][j] and bb[i][j]
        aa_addrs = aa_ptr + aa_base + j_indices
        bb_addrs = bb_ptr + bb_base + j_indices
        aa_vals = tl.load(aa_addrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_addrs, mask=j_mask, other=0.0)
        
        # Load cc[j][i] - note the transposed access
        cc_addrs = cc_ptr + j_indices * N + i
        cc_vals = tl.load(cc_addrs, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store back to aa[i][j]
        tl.store(aa_addrs, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1115_kernel[grid](aa, bb, cc, N, BLOCK_SIZE)