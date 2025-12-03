import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(n):
        # Zero out column i
        for j_start in range(0, n, BLOCK_SIZE):
            j_offsets = j_start + offsets
            j_mask = j_offsets < n
            
            # aa[j][i] = 0.0 (column i, rows j_offsets)
            store_offsets = j_offsets * n + i
            tl.store(aa_ptr + store_offsets, 0.0, mask=j_mask)
        
        # Set diagonal element aa[i][i] = 1.0
        diag_offset = i * n + i
        tl.store(aa_ptr + diag_offset, 1.0)

def s2102_triton(aa):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single block since we need sequential column processing
    grid = (1,)
    s2102_kernel[grid](aa, n, BLOCK_SIZE)
    
    return aa