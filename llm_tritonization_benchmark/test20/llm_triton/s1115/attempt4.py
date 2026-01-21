import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE_J: tl.constexpr):
    pid_j = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_idx = pid_j * BLOCK_SIZE_J + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # Load aa[i, j]
        aa_offset = i * N + j_idx
        aa_vals = tl.load(aa_ptr + aa_offset, mask=j_mask, other=0.0)
        
        # Load bb[i, j]
        bb_offset = i * N + j_idx
        bb_vals = tl.load(bb_ptr + bb_offset, mask=j_mask, other=0.0)
        
        # Load cc[j, i] 
        cc_offset = j_idx * N + i
        cc_vals = tl.load(cc_ptr + cc_offset, mask=j_mask, other=0.0)
        
        # Compute aa[i, j] = aa[i, j] * cc[j, i] + bb[i, j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store back to aa[i, j]
        tl.store(aa_ptr + aa_offset, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE_J = min(256, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE_J),)
    
    s1115_kernel[grid](aa, bb, cc, N, BLOCK_SIZE_J=BLOCK_SIZE_J)