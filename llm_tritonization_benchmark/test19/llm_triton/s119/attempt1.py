import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get current diagonal and position within diagonal
    diag = tl.program_id(0) + 2  # Start from diagonal 2 (i+j=2)
    pos = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate i and j from diagonal and position
    # For diagonal k, i ranges from max(1, k-N+1) to min(k-1, N-1)
    start_i = tl.maximum(1, diag - N + 1)
    end_i = tl.minimum(diag - 1, N - 1)
    
    i = start_i + pos
    j = diag - i
    
    # Check bounds
    mask = (i <= end_i) & (i >= 1) & (j >= 1) & (j < N)
    
    # Load values
    aa_idx = i * N + j
    bb_idx = i * N + j
    prev_aa_idx = (i - 1) * N + (j - 1)
    
    aa_prev = tl.load(aa_ptr + prev_aa_idx, mask=mask)
    bb_val = tl.load(bb_ptr + bb_idx, mask=mask)
    
    # Compute and store
    result = aa_prev + bb_val
    tl.store(aa_ptr + aa_idx, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):  # diag = i + j
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)
        
        if start_i <= end_i:
            num_elements = end_i - start_i + 1
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s119_kernel[grid](
                aa, bb, N, BLOCK_SIZE=BLOCK_SIZE
            )