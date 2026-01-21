import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Load b[i] and c[i], compute a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, N):
        # Load aa[j-1][i] for all i in this block
        aa_prev_offsets = (j - 1) * N + i_idx
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=i_mask)
        
        # Load bb[j][i] for all i in this block
        bb_offsets = j * N + i_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i] for all i in this block
        aa_offsets = j * N + i_idx
        tl.store(aa_ptr + aa_offsets, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s235_kernel[grid](a, aa, b, bb, c, N, BLOCK_SIZE=BLOCK_SIZE)