import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Get block start and create offset vector
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Create mask for valid indices (i must be > j and < N)
    mask = (i_indices > j) & (i_indices < N)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Load aa[j][i] values (row-major: aa[j][i] = aa[j*N + i])
    aa_ji_indices = j * N + i_indices
    aa_ji = tl.load(aa_ptr + aa_ji_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar, broadcast to vector)
    a_j_scalar = tl.load(a_ptr + j)
    a_j = tl.broadcast_to(a_j_scalar, [BLOCK_SIZE])
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel over i for each j
    for j in range(N):
        # Launch kernel with grid covering all possible i values
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)