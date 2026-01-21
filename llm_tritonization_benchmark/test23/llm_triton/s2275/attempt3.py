import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid_j * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    # Sequential loop over i dimension (in-kernel loop for dependencies)
    for i in range(len_2d):
        # For 2D arrays: aa[j][i] access pattern means row j, column i
        # In row-major layout: aa[j][i] = aa_ptr[j * len_2d + i]
        aa_indices = j_idx * len_2d + i
        
        # Load current values with masking
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + aa_indices, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + aa_indices, mask=j_mask, other=0.0)
        
        # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store result back
        tl.store(aa_ptr + aa_indices, result, mask=j_mask)
        
        # Handle 1D array computation for i-th element (only one thread does this)
        if pid_j == 0 and j_idx[0] == 0:
            # Load 1D array values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute: a[i] = b[i] + c[i] * d[i]
            a_result = b_val + c_val * d_val
            
            # Store result
            tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    # Get dimensions from input tensors
    len_2d = aa.shape[0]
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size for j dimension
    grid_j = triton.cdiv(len_2d, BLOCK_SIZE)
    
    # Launch kernel
    s2275_kernel[(grid_j,)](
        a, aa, b, bb, c, cc, d,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )