import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(
    a_ptr,
    aa_ptr,
    LEN_2D,
    j_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for back substitution - processes one column j at a time.
    Each thread block handles a range of rows i where i > j.
    """
    # Get thread block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate starting row index (must be > j_idx)
    i_start = j_idx + 1 + block_start
    
    # Create offsets for this block
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = (i_offsets < LEN_2D)
    
    # Load a[j] (scalar, same for all threads)
    a_j = tl.load(a_ptr + j_idx)
    
    # Load aa[j, i] values for this block
    aa_offsets = j_idx * LEN_2D + i_offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    a_i_new = a_i - aa_ji * a_j
    
    # Store back to memory
    tl.store(a_ptr + i_offsets, a_i_new, mask=mask)

def s115_triton(a, aa):
    """
    Triton implementation of TSVC s115 - back substitution.
    
    Args:
        a: 1D tensor (read-write)
        aa: 2D tensor (read-only)
    
    Returns:
        torch.Tensor: Modified array a
    """
    a = a.contiguous()
    aa = aa.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Block size for parallelization
    BLOCK_SIZE = 256
    
    # Process each column j sequentially (dependency constraint)
    for j in range(LEN_2D):
        # Number of elements to process for this j: i from j+1 to LEN_2D-1
        num_elements = LEN_2D - (j + 1)
        
        if num_elements <= 0:
            continue
            
        # Calculate grid size
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        if grid_size > 0:
            # Launch kernel for this column j
            s115_kernel[(grid_size,)](
                a,
                aa,
                LEN_2D,
                j,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    
    return a