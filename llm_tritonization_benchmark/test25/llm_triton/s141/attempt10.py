import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, n):
    i = tl.program_id(0)
    
    if i >= n:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i
    k = i * (i + 1) // 2 + i
    
    # Process all j values for this i
    for j in range(i, n):
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * n + i)
        
        # Load current flat_2d_array[k], add bb[j][i], store back
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration: k += (j+1)
        k += (j + 1)

def s141_triton(bb, flat_2d_array):
    n = bb.shape[0]
    
    # Use atomic operations to handle race conditions
    @triton.jit
    def s141_atomic_kernel(bb_ptr, flat_2d_array_ptr, n):
        i = tl.program_id(0)
        
        if i >= n:
            return
        
        # Calculate initial k value using the exact C formula
        # k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
        i_plus_1 = i + 1
        k = i_plus_1 * (i_plus_1 - 1) // 2 + i_plus_1 - 1
        
        # Process all j values for this i
        for j in range(i, n):
            # Load bb[j][i]
            bb_val = tl.load(bb_ptr + j * n + i)
            
            # Atomic add to flat_2d_array[k]
            tl.atomic_add(flat_2d_array_ptr + k, bb_val)
            
            # Update k for next iteration: k += (j+1)
            k += (j + 1)
    
    grid = (n,)
    s141_atomic_kernel[grid](bb, flat_2d_array, n)
    
    return flat_2d_array