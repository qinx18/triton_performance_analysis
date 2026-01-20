import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # Calculate offsets for aa[i][j] and bb[i][j]
        aa_offsets = i * N + j_idx
        bb_offsets = i * N + j_idx
        
        # Calculate offsets for cc[j][i] - note the transposed indices
        cc_offsets = j_idx * N + i
        
        # Load aa[i][j] and bb[i][j]
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # For cc[j][i], we need to load element by element since j_idx might be out of bounds
        cc_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            j = pid * BLOCK_SIZE + k
            if j < N:
                cc_offset = j * N + i
                cc_val = tl.load(cc_ptr + cc_offset)
                cc_vals = tl.where(j_offsets == k, cc_val, cc_vals)
        
        # Compute aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result back to aa[i][j]
        tl.store(aa_ptr + aa_offsets, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1115_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)