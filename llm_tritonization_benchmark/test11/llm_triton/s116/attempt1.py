import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(
    a_ptr,
    a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks of 5 elements at a time
    for i in range(0, BLOCK_SIZE, 5):
        base_idx = block_start + i
        
        # Check bounds for all 5 elements plus lookahead
        if base_idx + 5 >= n_elements:
            break
            
        # Load values from read-only copy
        idx_offsets = offsets[i:i+5] + block_start
        copy_offsets = tl.arange(0, 6) + base_idx  # Need 6 elements for lookahead
        
        mask = copy_offsets < n_elements
        a_vals = tl.load(a_copy_ptr + copy_offsets, mask=mask, other=0.0)
        
        # Main computation - S0 executes for all iterations
        if base_idx < n_elements - 5:
            # S0: a[i] = a[i + 1] * a[i]
            result0 = a_vals[1] * a_vals[0]
            tl.store(a_ptr + base_idx, result0)
        
        # Epilogue - overwritten statements only for last iterations
        # S1: execute only at last iteration (i = N-6)
        if base_idx == n_elements - 6:
            # S1: a[i + 1] = a[i + 2] * a[i + 1]
            result1 = a_vals[2] * a_vals[1]
            tl.store(a_ptr + base_idx + 1, result1)
        
        # S2: execute for last 2 iterations
        if base_idx >= n_elements - 7 and base_idx < n_elements - 5:
            # S2: a[i + 2] = a[i + 3] * a[i + 2]
            result2 = a_vals[3] * a_vals[2]
            tl.store(a_ptr + base_idx + 2, result2)
        
        # S3: execute for last 3 iterations  
        if base_idx >= n_elements - 8 and base_idx < n_elements - 5:
            # S3: a[i + 3] = a[i + 4] * a[i + 3]
            result3 = a_vals[4] * a_vals[3]
            tl.store(a_ptr + base_idx + 3, result3)
        
        # S4: execute for last 4 iterations
        if base_idx >= n_elements - 9 and base_idx < n_elements - 5:
            # S4: a[i + 4] = a[i + 5] * a[i + 4]
            result4 = a_vals[5] * a_vals[4]
            tl.store(a_ptr + base_idx + 4, result4)

def s116_triton(a):
    n_elements = a.shape[0] - 5
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s116_kernel[(grid_size,)](
        a,
        a_copy, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a