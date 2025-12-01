import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    j, k, LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for s132: aa[j][i] = aa[k][i-1] + b[i] * c[1]
    Uses coalesced memory access and vectorized operations
    """
    # Get program ID and compute indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (i >= 1 and i < LEN_2D)
    mask = (offsets >= 1) & (offsets < LEN_2D)
    
    # Load c[1] once (scalar broadcast)
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load aa[k][i-1] values with masking
    aa_k_indices = k * LEN_2D + (offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1
    
    # Store results to aa[j][i]
    aa_j_indices = j * LEN_2D + offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    """
    Triton implementation of TSVC s132
    Optimized with coalesced memory access and vectorized operations
    """
    aa = aa.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    LEN_2D = aa.shape[1]
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(grid_size,)](
        aa, b, c,
        j, k, LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa