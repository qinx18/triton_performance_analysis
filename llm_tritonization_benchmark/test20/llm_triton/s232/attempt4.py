import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < n
    
    for i in range(1, n):
        i_mask = i <= j_indices
        compute_mask = j_mask & i_mask
        
        # Load aa[j][i-1]
        prev_offsets = j_indices * n + (i - 1)
        prev_vals = tl.load(aa_ptr + prev_offsets, mask=compute_mask, other=0.0)
        
        # Load bb[j][i]
        bb_offsets = j_indices * n + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=compute_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
        result = prev_vals * prev_vals + bb_vals
        
        # Store aa[j][i]
        curr_offsets = j_indices * n + i
        tl.store(aa_ptr + curr_offsets, result, mask=compute_mask)

def s232_triton(aa, bb):
    n = aa.shape[0]
    BLOCK_SIZE = 64
    
    if n > 1:
        grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
        s232_kernel[grid](aa, bb, n, BLOCK_SIZE=BLOCK_SIZE)