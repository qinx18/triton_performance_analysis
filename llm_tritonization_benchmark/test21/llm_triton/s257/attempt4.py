import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load a[i-1]
    a_prev = tl.load(a_ptr + i_val - 1)
    
    # Process each j sequentially since they overwrite a[i]
    for j_start in range(0, N, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        current_j_mask = current_j_offsets < N
        
        # Load aa[j][i] and bb[j][i]
        aa_ptrs = aa_ptr + current_j_offsets * N + i_val
        bb_ptrs = bb_ptr + current_j_offsets * N + i_val
        
        aa_vals = tl.load(aa_ptrs, mask=current_j_mask)
        bb_vals = tl.load(bb_ptrs, mask=current_j_mask)
        
        # For each valid j, compute a[i] = aa[j][i] - a[i-1]
        # Each j overwrites a[i], so we process sequentially
        for j_local in range(BLOCK_SIZE):
            if j_start + j_local < N:
                # Get current a[i] value
                current_a = tl.load(a_ptr + i_val)
                
                # a[i] = aa[j][i] - a[i-1]
                j_idx = j_start + j_local
                aa_val = tl.load(aa_ptr + j_idx * N + i_val)
                new_a = aa_val - a_prev
                
                # Store new a[i]
                tl.store(a_ptr + i_val, new_a)
                
                # aa[j][i] = a[i] + bb[j][i]
                bb_val = tl.load(bb_ptr + j_idx * N + i_val)
                new_aa = new_a + bb_val
                tl.store(aa_ptr + j_idx * N + i_val, new_aa)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)