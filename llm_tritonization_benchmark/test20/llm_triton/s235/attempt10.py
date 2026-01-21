import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Update a[i] += b[i] * c[i]
    a_vals = a_vals + b_vals * c_vals
    
    # Store updated a[i]
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j from 1 to n-1
    for j in range(1, n):
        # Load aa[j-1][i]
        prev_j_offsets = (j - 1) * n + i_idx
        aa_prev_vals = tl.load(aa_ptr + prev_j_offsets, mask=i_mask, other=0.0)
        
        # Load bb[j][i]
        curr_j_offsets = j * n + i_idx
        bb_vals = tl.load(bb_ptr + curr_j_offsets, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + curr_j_offsets, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )