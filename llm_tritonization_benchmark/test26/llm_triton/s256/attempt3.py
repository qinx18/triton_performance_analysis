import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j_val, i_start, N, BLOCK_SIZE: tl.constexpr):
    # Get block of i indices starting from i_start
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_start + i_offsets
    i_mask = i_indices < N
    
    # Load bb[j_val][i] values
    bb_offsets = j_val * N + i_indices
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
    
    # Load d[j_val] value (scalar broadcast)
    d_val = tl.load(d_ptr + j_val)
    
    # Load a[j_val] value (scalar broadcast)
    a_val = tl.load(a_ptr + j_val)
    
    # Compute aa[j_val][i] = a[j_val] + bb[j_val][i] * d[j_val]
    aa_new = a_val + bb_vals * d_val
    
    # Store aa[j_val][i] values
    aa_offsets = j_val * N + i_indices
    tl.store(aa_ptr + aa_offsets, aa_new, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel processing of i
    for j in range(1, N):
        # Update a[j] = 1.0 - a[j-1]
        a[j] = 1.0 - a[j-1]
        
        # Launch kernels for all i values in parallel
        num_blocks = triton.cdiv(N, BLOCK_SIZE)
        grid = (num_blocks,)
        
        for block_idx in range(num_blocks):
            i_start = block_idx * BLOCK_SIZE
            if i_start < N:
                s256_kernel[grid](
                    a, aa, bb, d,
                    j, i_start, N, BLOCK_SIZE
                )