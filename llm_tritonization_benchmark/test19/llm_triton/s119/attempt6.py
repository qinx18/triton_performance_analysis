import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, diag: tl.constexpr, start_i: tl.constexpr, end_i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i = start_i + pid
    if i >= end_i:
        return
    
    j = diag - i
    if j < 1 or j >= N:
        return
        
    # Load aa[i-1][j-1] + bb[i][j]
    aa_idx = (i - 1) * N + (j - 1)
    bb_idx = i * N + j
    out_idx = i * N + j
    
    aa_val = tl.load(aa_ptr + aa_idx)
    bb_val = tl.load(bb_ptr + bb_idx)
    result = aa_val + bb_val
    
    tl.store(aa_ptr + out_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):  # diag = i + j
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        BLOCK_SIZE = 128
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, N, diag, start_i, end_i, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa