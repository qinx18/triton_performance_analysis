import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    i_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Valid indices are j+1 <= i < N
    mask = (i_offset >= j + 1) & (i_offset < N)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offset, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values (aa is row-major: aa[j][i] = aa[j*N + i])
    aa_ji_offset = j * N + i_offset
    aa_ji = tl.load(aa_ptr + aa_ji_offset, mask=mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offset, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(N):
        # Calculate number of valid i indices (from j+1 to N-1)
        num_valid = N - (j + 1)
        
        # Only launch kernel if there are valid i indices
        if num_valid > 0:
            grid = (triton.cdiv(num_valid, BLOCK_SIZE),)
            s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)