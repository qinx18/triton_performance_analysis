import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements - 5, 5):
        # Only process this iteration if it's assigned to this block
        process_iteration = (block_start == 0)
        
        if process_iteration & (i + 5 < n_elements):
            # Load values from read-only copy
            a_i = tl.load(a_copy_ptr + i)
            a_i1 = tl.load(a_copy_ptr + i + 1)
            a_i2 = tl.load(a_copy_ptr + i + 2)
            a_i3 = tl.load(a_copy_ptr + i + 3)
            a_i4 = tl.load(a_copy_ptr + i + 4)
            a_i5 = tl.load(a_copy_ptr + i + 5)
            
            # Compute and store to original array
            tl.store(a_ptr + i, a_i1 * a_i)
            tl.store(a_ptr + i + 1, a_i2 * a_i1)
            tl.store(a_ptr + i + 2, a_i3 * a_i2)
            tl.store(a_ptr + i + 3, a_i4 * a_i3)
            tl.store(a_ptr + i + 4, a_i5 * a_i4)

def s116_triton(a):
    n_elements = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 1
    grid = (1,)
    
    s116_kernel[grid](a, a_copy, n_elements, BLOCK_SIZE)