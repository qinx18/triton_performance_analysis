import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE_J: tl.constexpr):
    pid_j = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_indices = pid_j * BLOCK_SIZE_J + j_offsets
    j_mask = j_indices < N
    
    for i in range(N):
        # Calculate base addresses for current i
        aa_base = aa_ptr + i * N
        bb_base = bb_ptr + i * N
        
        # Load aa[i][j] and bb[i][j]
        aa_vals = tl.load(aa_base + j_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_base + j_indices, mask=j_mask, other=0.0)
        
        # Load cc[j][i] - note the transposed access
        cc_vals = tl.zeros([BLOCK_SIZE_J], dtype=tl.float32)
        for idx in range(BLOCK_SIZE_J):
            if j_indices[idx] < N:
                cc_addr = cc_ptr + j_indices[idx] * N + i
                cc_vals = tl.where(idx == tl.arange(0, BLOCK_SIZE_J), 
                                 tl.load(cc_addr), cc_vals)
        
        # Compute: aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result
        tl.store(aa_base + j_indices, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    
    BLOCK_SIZE_J = 64
    grid = (triton.cdiv(N, BLOCK_SIZE_J),)
    
    s1115_kernel[grid](
        aa, bb, cc, N,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )