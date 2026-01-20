import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, diag: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i from diagonal and pid
    start_i = tl.maximum(1, diag - N + 1)
    i = start_i + pid
    
    # Check bounds
    if i >= tl.minimum(diag, N):
        return
    
    j = diag - i
    
    # Check j bounds
    if j < 1:
        return
    if j >= N:
        return
    
    # Load values
    aa_prev_idx = (i - 1) * N + (j - 1)
    bb_idx = i * N + j
    out_idx = i * N + j
    
    aa_prev = tl.load(aa_ptr + aa_prev_idx)
    bb_val = tl.load(bb_ptr + bb_idx)
    result = aa_prev + bb_val
    
    tl.store(aa_ptr + out_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        num_elements = end_i - start_i
        if num_elements <= 0:
            continue
        
        BLOCK_SIZE = 64
        grid = (num_elements,)
        
        s119_kernel[grid](
            aa, bb, N, diag, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa