import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Compute k = 2*i
        k = 2 * idx
        
        # Check bounds for all accesses
        if k + 1 < n_elements * 2:  # b array has full size
            # Load values
            d_val = tl.load(d_ptr + idx)
            b_val = tl.load(b_ptr + k)
            c_val = tl.load(c_ptr + k)
            
            # Compute a[i] = b[k] - d[i]
            a_val = b_val - d_val
            tl.store(a_ptr + idx, a_val)
            
            # Compute b[k] = a[i] + c[k]
            b_new_val = a_val + c_val
            tl.store(b_ptr + k, b_new_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s128_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )