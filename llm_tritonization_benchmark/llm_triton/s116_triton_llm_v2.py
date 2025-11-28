import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(
    a_ptr,
    a_copy_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process in chunks of 5, so we need to ensure we have enough elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n - 5
    
    # We need to process in steps of 5
    for step in range(0, BLOCK_SIZE, 5):
        i = block_start + step
        if i >= n - 5:
            break
            
        # Load values from the read-only copy
        a_i = tl.load(a_copy_ptr + i, mask=i < n)
        a_i1 = tl.load(a_copy_ptr + i + 1, mask=i + 1 < n)
        a_i2 = tl.load(a_copy_ptr + i + 2, mask=i + 2 < n)
        a_i3 = tl.load(a_copy_ptr + i + 3, mask=i + 3 < n)
        a_i4 = tl.load(a_copy_ptr + i + 4, mask=i + 4 < n)
        a_i5 = tl.load(a_copy_ptr + i + 5, mask=i + 5 < n)
        
        # Compute the new values
        temp0 = a_i1 * a_i
        temp1 = a_i2 * a_i1
        temp2 = a_i3 * a_i2
        temp3 = a_i4 * a_i3
        temp4 = a_i5 * a_i4
        
        # Store results to the original array
        tl.store(a_ptr + i, temp0, mask=i < n)
        tl.store(a_ptr + i + 1, temp1, mask=i + 1 < n)
        tl.store(a_ptr + i + 2, temp2, mask=i + 2 < n)
        tl.store(a_ptr + i + 3, temp3, mask=i + 3 < n)
        tl.store(a_ptr + i + 4, temp4, mask=i + 4 < n)

def s116_triton(a):
    a = a.contiguous()
    n = a.size(0)
    
    if n <= 5:
        return a
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate grid size - process in chunks of 5, but use larger blocks
    BLOCK_SIZE = 256
    # Ensure we cover all elements that need processing (n-5 elements in groups of 5)
    num_elements_to_process = ((n - 5) // 5) * 5
    grid = (triton.cdiv(num_elements_to_process, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a