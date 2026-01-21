import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[i-1] once
    a_prev = tl.load(a_ptr + i_val - 1)
    
    # Process all j values in blocks
    for j_start in range(0, N, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        j_mask = current_j_offsets < N
        
        # Load aa[j][i] and bb[j][i] for current block
        aa_ptrs = aa_ptr + current_j_offsets * N + i_val
        bb_ptrs = bb_ptr + current_j_offsets * N + i_val
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1] for each j
        # Since all j iterations overwrite a[i], we need the last valid j
        for local_j in range(BLOCK_SIZE):
            if j_start + local_j < N:
                # Extract scalar value for this j
                aa_val = tl.load(aa_ptr + (j_start + local_j) * N + i_val)
                
                # a[i] = aa[j][i] - a[i-1]
                new_a = aa_val - a_prev
                tl.store(a_ptr + i_val, new_a)
                
                # aa[j][i] = a[i] + bb[j][i]
                bb_val = tl.load(bb_ptr + (j_start + local_j) * N + i_val)
                new_aa = new_a + bb_val
                tl.store(aa_ptr + (j_start + local_j) * N + i_val, new_aa)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)