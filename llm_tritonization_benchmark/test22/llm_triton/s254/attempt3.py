import torch
import triton
import triton.language as tl

@triton.jit
def s254_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Only process with first thread to maintain sequential dependency
    if pid == 0:
        # Process all elements sequentially
        x_val = tl.load(b_ptr + (N - 1))  # x = b[LEN_1D-1]
        
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            # Load b values for this block
            b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
            
            # Compute a[i] = (b[i] + x) * 0.5 for each element in block
            results = tl.zeros_like(b_vals)
            
            # Process each element in the block
            for local_i in range(BLOCK_SIZE):
                global_i = block_start + local_i
                if global_i < N:
                    b_i = tl.load(b_ptr + global_i)
                    result = (b_i + x_val) * 0.5
                    tl.store(a_ptr + global_i, result)
                    x_val = b_i  # x = b[i]

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single thread to maintain sequential dependency
    s254_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)