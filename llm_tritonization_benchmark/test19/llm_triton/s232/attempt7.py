import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    j_block_id = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_block_id * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < N
    
    for i in range(1, N):
        # Mask for valid j values where i <= j
        valid_mask = j_mask & (i <= j_indices)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Load aa[j][i-1]
            aa_prev_ptrs = aa_ptr + j_indices * N + (i - 1)
            aa_prev_vals = tl.load(aa_prev_ptrs, mask=valid_mask, other=0.0)
            
            # Load bb[j][i]
            bb_ptrs = bb_ptr + j_indices * N + i
            bb_vals = tl.load(bb_ptrs, mask=valid_mask, other=0.0)
            
            # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
            new_vals = aa_prev_vals * aa_prev_vals + bb_vals
            
            # Store aa[j][i]
            aa_curr_ptrs = aa_ptr + j_indices * N + i
            tl.store(aa_curr_ptrs, new_vals, mask=valid_mask)

def s232_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s232_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)