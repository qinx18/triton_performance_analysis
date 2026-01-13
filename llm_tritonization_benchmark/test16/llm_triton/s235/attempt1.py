import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j
    for j in range(1, N):
        # Load aa[j-1][i] (previous row)
        aa_prev_ptrs = aa_ptr + (j - 1) * N + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j * N + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j * N + i_idx
        tl.store(aa_ptrs, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s235_kernel[grid](
        a, aa, b, bb, c, N, BLOCK_SIZE
    )