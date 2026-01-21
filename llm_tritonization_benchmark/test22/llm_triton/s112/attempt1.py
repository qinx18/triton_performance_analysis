import triton
import triton.language as tl

@triton.jit
def s112_kernel(a_ptr, a_copy_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+1] (original array)
    output_indices = indices + 1
    output_mask = output_indices < (n + 1)
    tl.store(a_ptr + output_indices, result, mask=output_mask)

def s112_triton(a, b):
    n = a.shape[0] - 1  # Loop from LEN_1D - 2 to 0, so n = LEN_1D - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a, a_copy, b, n, BLOCK_SIZE=BLOCK_SIZE
    )