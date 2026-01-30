import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_start, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID and create block of i indices
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + i_start
    
    # Mask for valid i indices
    i_mask = i_indices < len_2d
    
    # Load bb[j][i] and d[j] values - j is fixed for this kernel launch
    bb_indices = i_indices  # bb is passed as bb[j] slice
    bb_vals = tl.load(bb_ptr + bb_indices, mask=i_mask, other=0.0)
    
    # Load a[j] value (scalar broadcast)
    a_val = tl.load(a_ptr)
    
    # Load d[j] value (scalar)
    d_val = tl.load(d_ptr)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] values
    tl.store(aa_ptr + bb_indices, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 128
    
    # Process each i sequentially
    for i in range(len_2d):
        # Process each j sequentially (j dependency requires this)
        for j in range(1, len_2d):
            # Update a[j] = 1.0 - a[j-1]
            a[j] = 1.0 - a[j-1]
            
            # Calculate grid size for i dimension
            grid_size = triton.cdiv(len_2d, BLOCK_SIZE)
            
            # Launch kernel for this j value, parallelizing over i
            s256_kernel[(grid_size,)](
                a[j:j+1],  # Pass a[j] as single element
                aa[j:j+1, :],  # Pass aa[j][:] row
                bb[j:j+1, :],  # Pass bb[j][:] row  
                d[j:j+1],  # Pass d[j] as single element
                0,  # i_start
                len_2d, 
                BLOCK_SIZE
            )