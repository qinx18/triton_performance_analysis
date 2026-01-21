import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, N, i, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = current_j < N
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + current_j * N + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        # Load aa[j][i]
        aa_ptrs = aa_ptr + current_j * N + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        
        # Load a[i-1]
        a_prev = tl.load(a_ptr + i - 1)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Store aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        tl.store(aa_ptrs, aa_new, mask=j_mask)
        
        # Store a[i] - last j iteration will overwrite
        tl.store(a_ptr + i, a_new)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, N, i, BLOCK_SIZE=BLOCK_SIZE)