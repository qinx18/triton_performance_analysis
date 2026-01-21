import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, start_i, end_i, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which i value this block handles
    i = start_i + pid
    
    if i >= end_i:
        return
    
    # For this i, j = diag - i
    j = diag - i
    
    # Bounds check
    if i >= N or j >= N or i < 1 or j < 1:
        return
    
    # Calculate offsets
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
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
        
        # Number of elements on this diagonal
        num_elements = end_i - start_i
        
        # Launch kernel with one thread per element on this diagonal
        grid = (num_elements,)
        s119_kernel[grid](aa, bb, diag, start_i, end_i, N=N, BLOCK_SIZE=256)
    
    return aa