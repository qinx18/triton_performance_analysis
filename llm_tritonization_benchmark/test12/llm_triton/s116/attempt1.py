import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S0 for all valid iterations
    for iter_start in range(0, n_elements - 4, BLOCK_SIZE):
        current_offsets = iter_start + offsets
        mask = (current_offsets < n_elements - 4) & (current_offsets < iter_start + BLOCK_SIZE)
        
        if pid == iter_start // BLOCK_SIZE:
            # S0: a[i] = a[i + 1] * a[i]
            a_i = tl.load(a_copy_ptr + current_offsets, mask=mask)
            a_i_plus_1 = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
            result = a_i_plus_1 * a_i
            tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Epilogue for overwritten statements
    if pid == 0:
        # S1: execute only at i = n-5 (last iteration where i+1 < n_elements)
        if n_elements >= 6:
            i = n_elements - 5
            a_i_plus_1 = tl.load(a_copy_ptr + i + 1)
            a_i_plus_2 = tl.load(a_copy_ptr + i + 2)
            result = a_i_plus_2 * a_i_plus_1
            tl.store(a_ptr + i + 1, result)
        
        # S2: execute for last 2 iterations
        for offset in range(max(0, n_elements - 6), n_elements - 4):
            if offset + 2 < n_elements:
                a_i_plus_2 = tl.load(a_copy_ptr + offset + 2)
                a_i_plus_3 = tl.load(a_copy_ptr + offset + 3)
                result = a_i_plus_3 * a_i_plus_2
                tl.store(a_ptr + offset + 2, result)
        
        # S3: execute for last 3 iterations
        for offset in range(max(0, n_elements - 7), n_elements - 4):
            if offset + 3 < n_elements:
                a_i_plus_3 = tl.load(a_copy_ptr + offset + 3)
                a_i_plus_4 = tl.load(a_copy_ptr + offset + 4)
                result = a_i_plus_4 * a_i_plus_3
                tl.store(a_ptr + offset + 3, result)
        
        # S4: execute for last 4 iterations
        for offset in range(max(0, n_elements - 8), n_elements - 4):
            if offset + 4 < n_elements:
                a_i_plus_4 = tl.load(a_copy_ptr + offset + 4)
                a_i_plus_5 = tl.load(a_copy_ptr + offset + 5)
                result = a_i_plus_5 * a_i_plus_4
                tl.store(a_ptr + offset + 4, result)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy BEFORE launching kernel
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Pass BOTH original (for writes) AND copy (for reads) to kernel
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )