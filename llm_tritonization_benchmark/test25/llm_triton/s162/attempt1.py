import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load from read-only copy at offset positions
        read_offsets = current_offsets + k
        read_mask = mask & (read_offsets < (n + k))
        a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        
        # Load b and c at current positions
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute result
        result = a_vals + b_vals * c_vals
        
        # Store to original array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    n = a.shape[0] - 1  # LEN_1D - 1
    
    if k > 0:
        # Create read-only copy to handle WAR dependencies
        a_copy = a.clone()
        
        BLOCK_SIZE = 256
        grid = (1,)
        
        s162_kernel[grid](
            a, a_copy, b, c, n, k,
            BLOCK_SIZE=BLOCK_SIZE
        )