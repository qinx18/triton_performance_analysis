import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements - 5
    
    # S0: a[i] = a[i + 1] * a[i] for all valid iterations
    a_i = tl.load(a_copy_ptr + offsets, mask=mask)
    a_i_plus_1 = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    result = a_i_plus_1 * a_i
    tl.store(a_ptr + offsets, result, mask=mask)
    
    # Epilogue for overwritten statements - only execute on first block
    if pid == 0:
        # S1: execute only for last 1 iteration (n-6)
        last_iter = n_elements - 6
        if last_iter >= 0:
            a_val = tl.load(a_copy_ptr + last_iter + 2)
            a_val_next = tl.load(a_copy_ptr + last_iter + 1)  
            result_s1 = a_val * a_val_next
            tl.store(a_ptr + last_iter + 1, result_s1)
        
        # S2: execute for last 2 iterations (n-7, n-6)
        for offset in range(max(0, n_elements - 7), n_elements - 5):
            if offset + 2 < n_elements:
                a_val = tl.load(a_copy_ptr + offset + 3)
                a_val_prev = tl.load(a_copy_ptr + offset + 2)
                result_s2 = a_val * a_val_prev
                tl.store(a_ptr + offset + 2, result_s2)
        
        # S3: execute for last 3 iterations (n-8, n-7, n-6)
        for offset in range(max(0, n_elements - 8), n_elements - 5):
            if offset + 3 < n_elements:
                a_val = tl.load(a_copy_ptr + offset + 4)
                a_val_prev = tl.load(a_copy_ptr + offset + 3)
                result_s3 = a_val * a_val_prev
                tl.store(a_ptr + offset + 3, result_s3)
        
        # S4: execute for last 4 iterations (n-9, n-8, n-7, n-6)
        for offset in range(max(0, n_elements - 9), n_elements - 5):
            if offset + 4 < n_elements:
                a_val = tl.load(a_copy_ptr + offset + 5)
                a_val_prev = tl.load(a_copy_ptr + offset + 4)
                result_s4 = a_val * a_val_prev
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