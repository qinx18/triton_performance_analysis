import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(
    a_ptr,
    aa_ptr,
    j,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + (j + 1)
    
    # Create mask for valid i indices (i < N)
    mask = i_offsets < N
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    
    # Load aa[j][i] values (aa is row-major: aa[j][i] = aa[j*N + i])
    aa_ji_offsets = j * N + i_offsets
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(N):
        # Calculate number of i values from j+1 to N-1
        num_i_values = N - (j + 1)
        if num_i_values <= 0:
            continue
            
        num_blocks = triton.cdiv(num_i_values, BLOCK_SIZE)
        
        # Launch kernel for this j with i-parallel
        grid = (num_blocks,)
        s115_kernel[grid](
            a,
            aa,
            j,
            N,
            BLOCK_SIZE,
        )