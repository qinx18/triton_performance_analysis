import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        if i == 0:
            pass
        else:
            j_base = pid * BLOCK_SIZE
            j_idx = j_base + j_offsets
            
            # Mask for valid j indices and triangular constraint j < i
            mask = (j_idx < i) & (j_idx >= 0)
            
            if tl.max(mask.to(tl.int32)) > 0:
                # Calculate linear indices for aa[i][j]
                aa_write_idx = i * N + j_idx
                # Calculate linear indices for aa[j][i] 
                aa_read_idx = j_idx * N + i
                # Calculate linear indices for bb[i][j]
                bb_idx = i * N + j_idx
                
                # Load values
                aa_ji = tl.load(aa_ptr + aa_read_idx, mask=mask, other=0.0)
                bb_ij = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
                
                # Compute result
                result = aa_ji + bb_ij
                
                # Store result
                tl.store(aa_ptr + aa_write_idx, result, mask=mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate maximum number of j values (triangular: max i-1 values)
    max_j = N - 1
    grid = (triton.cdiv(max_j, BLOCK_SIZE),)
    
    s114_kernel[grid](aa, bb, N=N, BLOCK_SIZE=BLOCK_SIZE)