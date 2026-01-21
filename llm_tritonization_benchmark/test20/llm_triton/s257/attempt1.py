import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j = j_start + j_offsets
        valid_j = (current_j < N) & j_mask
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + current_j * N + i
        bb_vals = tl.load(bb_ptrs, mask=valid_j)
        
        # Load aa[j][i]
        aa_ptrs = aa_ptr + current_j * N + i
        aa_vals = tl.load(aa_ptrs, mask=valid_j)
        
        # Load a[i-1]
        a_prev = tl.load(a_ptr + i - 1)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        # Since all j iterations overwrite a[i], we need the final result
        # We'll compute for all j values and use the last valid one
        a_new = aa_vals - a_prev
        
        # Store aa[j][i] = a[i] + bb[j][i] for all valid j
        aa_new = a_new + bb_vals
        tl.store(aa_ptrs, aa_new, mask=valid_j)
        
        # For a[i], we need to store the value from the last j iteration
        # Find the last valid j index
        if j_start + BLOCK_SIZE >= N:
            # This is the last block, find the actual last j
            last_j_in_block = (N - 1) % BLOCK_SIZE
            if tl.program_id(0) == 0:  # Only one thread stores a[i]
                final_a = tl.load(a_ptr + i - 1)
                # We need to recompute with the last j value
                last_j = N - 1
                aa_last_ptr = aa_ptr + last_j * N + i
                aa_last_val = tl.load(aa_last_ptr)
                final_a_val = aa_last_val - final_a
                tl.store(a_ptr + i, final_a_val)

def s257_triton(a, aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, N, i=i, BLOCK_SIZE=BLOCK_SIZE)