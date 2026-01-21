import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    j_idx = j_base + j_offsets
    j_mask = j_idx < N
    
    for i in range(1, N):
        # Previous row indices (i-1)
        prev_row_idx = (i - 1) * N + j_idx
        # Current row indices (i)
        current_row_idx = i * N + j_idx
        
        # Load aa[i-1][j]
        aa_prev = tl.load(aa_ptr + prev_row_idx, mask=j_mask, other=0.0)
        # Load bb[i][j]
        bb_curr = tl.load(bb_ptr + current_row_idx, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_curr
        
        # Store result to aa[i][j]
        tl.store(aa_ptr + current_row_idx, result, mask=j_mask)

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)