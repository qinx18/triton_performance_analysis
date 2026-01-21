import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < N
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=mask_i, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=mask_i, other=0.0)
    
    # Update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i, other=0.0)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)
    
    # Sequential loop over j
    for j in range(1, N):
        # Load aa[j-1][i]
        prev_idx = (j - 1) * N + i_idx
        aa_prev = tl.load(aa_ptr + prev_idx, mask=mask_i, other=0.0)
        
        # Load bb[j][i]
        bb_idx = j * N + i_idx
        bb_vals = tl.load(bb_ptr + bb_idx, mask=mask_i, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new = aa_prev + bb_vals * a_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + bb_idx, aa_new, mask=mask_i)

def s235_triton(a, aa, b, bb, c):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s235_kernel[grid](a, aa, b, bb, c, N, BLOCK_SIZE=BLOCK_SIZE)