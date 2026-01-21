import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, N, i, a_prev, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_start = tl.program_id(0) * BLOCK_SIZE
    j = j_start + j_offsets
    j_mask = j < N
    
    # Load bb[j][i]
    bb_ptrs = bb_ptr + j * N + i
    bb_vals = tl.load(bb_ptrs, mask=j_mask)
    
    # Load aa[j][i]
    aa_ptrs = aa_ptr + j * N + i
    aa_vals = tl.load(aa_ptrs, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1]
    a_new = aa_vals - a_prev
    
    # Store aa[j][i] = a[i] + bb[j][i]
    aa_new = a_new + bb_vals
    tl.store(aa_ptrs, aa_new, mask=j_mask)
    
    # Store a[i] for each j (only last valid j matters)
    a_ptr_offset = a_ptr + i
    tl.store(a_ptr_offset, a_new, mask=j_mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        # Load a[i-1] once per i iteration
        a_prev = a[i-1].item()
        
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, N, i, a_prev, BLOCK_SIZE=BLOCK_SIZE)