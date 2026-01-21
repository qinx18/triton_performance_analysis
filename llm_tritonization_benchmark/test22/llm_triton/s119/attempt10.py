import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, diag, start_i, end_i, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i = start_i + pid
    
    if (i >= 1) & (i < N) & (i <= end_i):
        j = diag - i
        
        if (j >= 1) & (j < N):
            current_idx = i * N + j
            prev_idx = (i - 1) * N + (j - 1)
            
            aa_prev = tl.load(aa_ptr + prev_idx)
            bb_val = tl.load(bb_ptr + current_idx)
            
            result = aa_prev + bb_val
            tl.store(aa_ptr + current_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)
        
        if start_i <= end_i:
            num_elements = end_i - start_i + 1
            grid = (triton.cdiv(num_elements, 1),)
            
            BLOCK_SIZE = 256
            
            s119_kernel[grid](
                aa, bb, N, diag, start_i, end_i, BLOCK_SIZE
            )
    
    return aa