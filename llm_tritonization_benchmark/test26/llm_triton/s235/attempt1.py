import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # First compute: a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    
    a_new = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_new, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, N):
        # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        prev_j_idx = (j - 1) * N + i_idx
        curr_j_idx = j * N + i_idx
        
        aa_prev = tl.load(aa_ptr + prev_j_idx, mask=i_mask)
        bb_curr = tl.load(bb_ptr + curr_j_idx, mask=i_mask)
        
        aa_new = aa_prev + bb_curr * a_new
        tl.store(aa_ptr + curr_j_idx, aa_new, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        N, BLOCK_SIZE
    )