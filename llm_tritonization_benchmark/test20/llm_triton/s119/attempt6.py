import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr):
    # Sequential processing due to wavefront dependencies
    for i in range(1, N):
        for j in range(1, N):
            # Calculate offsets for current position
            current_offset = i * N + j
            prev_diag_offset = (i - 1) * N + (j - 1)
            
            # Load values
            aa_prev = tl.load(aa_ptr + prev_diag_offset)
            bb_curr = tl.load(bb_ptr + current_offset)
            
            # Compute and store
            result = aa_prev + bb_curr
            tl.store(aa_ptr + current_offset, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Launch single thread to handle sequential computation
    grid = (1,)
    s119_kernel[grid](aa, bb, N=N)
    
    return aa