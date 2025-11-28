import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    j,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block start position for i dimension
    block_start = tl.program_id(0) * BLOCK_SIZE + (j + 1)
    
    # Create offsets for the i dimension (j+1 to LEN_2D-1)
    i_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid i values (i < LEN_2D)
    mask = i_offsets < LEN_2D
    
    # Load a[j] (broadcast value)
    a_j = tl.load(a_copy_ptr + j)
    
    # Calculate 2D indices for aa[j][i]
    aa_offsets = j * LEN_2D + i_offsets
    
    # Load aa[j][i] values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values from copy
    a_vals = tl.load(a_copy_ptr + i_offsets, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store result back to original array
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Sequential execution over j dimension
    for j in range(LEN_2D):
        # Calculate number of elements to process (j+1 to LEN_2D-1)
        num_elements = LEN_2D - (j + 1)
        
        if num_elements <= 0:
            continue
            
        # Calculate grid size for i dimension
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        if grid_size > 0:
            grid = (grid_size,)
            
            s115_kernel[grid](
                a,
                a_copy,
                aa,
                j,
                LEN_2D,
                BLOCK_SIZE,
            )
        
        # Update copy for next j iteration
        a_copy = a.clone()