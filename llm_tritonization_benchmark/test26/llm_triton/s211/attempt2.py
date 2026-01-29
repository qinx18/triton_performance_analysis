import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the reordered statements sequentially
    # due to the loop-carried dependency
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequential iterations within this block
    for local_i in range(BLOCK_SIZE):
        i = block_start + local_i
        
        # Check bounds: we need i from 1 to N-2 for the main loop
        if i < 1:
            pass
        elif i >= N - 1:
            pass
        else:
            # Handle prologue: first consumer uses original b[0]
            if i == 1:
                b_val_prev = tl.load(b_copy_ptr + 0)  # b[0]
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                a_val = b_val_prev + c_val * d_val
                tl.store(a_ptr + i, a_val)
                
            # Main loop body: producer first, then shifted consumer
            if i < N - 2:
                # Producer: b[i] = b_copy[i+1] - e[i] * d[i]
                b_val_next = tl.load(b_copy_ptr + i + 1)
                e_val = tl.load(e_ptr + i)
                d_val = tl.load(d_ptr + i)
                b_val = b_val_next - e_val * d_val
                tl.store(b_ptr + i, b_val)
                
                # Consumer shifted: a[i+1] = b[i] + c[i+1] * d[i+1]
                if i + 1 < N - 1:
                    c_val_next = tl.load(c_ptr + i + 1)
                    d_val_next = tl.load(d_ptr + i + 1)
                    a_val_next = b_val + c_val_next * d_val_next
                    tl.store(a_ptr + i + 1, a_val_next)
            
            # Epilogue: last producer has no consumer
            if i == N - 2:
                b_val_next = tl.load(b_copy_ptr + i + 1)
                e_val = tl.load(e_ptr + i)
                d_val = tl.load(d_ptr + i)
                b_val = b_val_next - e_val * d_val
                tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    b_copy = b.clone()
    
    # Use small block size due to sequential nature
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e, N, BLOCK_SIZE
    )