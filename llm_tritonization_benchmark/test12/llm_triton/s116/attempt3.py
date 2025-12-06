import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # S0: a[i] = a[i + 1] * a[i] for iterations 0 to n-6
    mask_s0 = (offsets < n_elements - 5) & ((offsets % 5) == 0)
    a_i = tl.load(a_copy_ptr + offsets, mask=mask_s0)
    a_i_plus_1 = tl.load(a_copy_ptr + offsets + 1, mask=mask_s0)
    result_s0 = a_i_plus_1 * a_i
    tl.store(a_ptr + offsets, result_s0, mask=mask_s0)
    
    # Epilogue for overwritten statements - only execute on first block
    if pid == 0:
        # S1: a[i+1] = a[i+2] * a[i+1] - execute for last 1 iteration only
        last_base = ((n_elements - 6) // 5) * 5
        if last_base >= 0:
            i_val = last_base
            if i_val + 1 < n_elements & i_val + 2 < n_elements:
                a_i_plus_1 = tl.load(a_copy_ptr + i_val + 1)
                a_i_plus_2 = tl.load(a_copy_ptr + i_val + 2)
                result_s1 = a_i_plus_2 * a_i_plus_1
                tl.store(a_ptr + i_val + 1, result_s1)
        
        # S2: a[i+2] = a[i+3] * a[i+2] - execute for last 2 iterations
        for offset in range(max(0, last_base - 5), last_base + 1, 5):
            if offset >= 0 & offset + 2 < n_elements & offset + 3 < n_elements:
                a_i_plus_2 = tl.load(a_copy_ptr + offset + 2)
                a_i_plus_3 = tl.load(a_copy_ptr + offset + 3)
                result_s2 = a_i_plus_3 * a_i_plus_2
                tl.store(a_ptr + offset + 2, result_s2)
        
        # S3: a[i+3] = a[i+4] * a[i+3] - execute for last 3 iterations
        for offset in range(max(0, last_base - 10), last_base + 1, 5):
            if offset >= 0 & offset + 3 < n_elements & offset + 4 < n_elements:
                a_i_plus_3 = tl.load(a_copy_ptr + offset + 3)
                a_i_plus_4 = tl.load(a_copy_ptr + offset + 4)
                result_s3 = a_i_plus_4 * a_i_plus_3
                tl.store(a_ptr + offset + 3, result_s3)
        
        # S4: a[i+4] = a[i+5] * a[i+4] - execute for last 4 iterations
        for offset in range(max(0, last_base - 15), last_base + 1, 5):
            if offset >= 0 & offset + 4 < n_elements & offset + 5 < n_elements:
                a_i_plus_4 = tl.load(a_copy_ptr + offset + 4)
                a_i_plus_5 = tl.load(a_copy_ptr + offset + 5)
                result_s4 = a_i_plus_5 * a_i_plus_4
                tl.store(a_ptr + offset + 4, result_s4)

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