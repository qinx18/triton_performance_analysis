import triton
import triton.language as tl

@triton.jit
def s162_kernel(
    a_ptr,
    a_copy_ptr, 
    b_ptr,
    c_ptr,
    k,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx < n_elements) & (k > 0)
    
    # Load from read-only copy with offset k
    a_offset_idx = idx + k
    a_vals = tl.load(a_copy_ptr + a_offset_idx, mask=mask)
    
    # Load b and c arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute: a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s162_triton(a, b, c, k):
    n_elements = a.shape[0] - 1  # LEN_1D-1 from the loop condition
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a,
        a_copy,
        b, 
        c,
        k,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )