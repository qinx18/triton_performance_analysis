import triton
import triton.language as tl

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M, N, A_stride0, A_stride1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize y to zero
        y_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_y_offsets = block_start + y_offsets
            y_mask = current_y_offsets < N
            tl.store(y_ptr + current_y_offsets, 0.0, mask=y_mask)
    
    tl.debug_barrier()
    
    # Each program handles one row of A
    i = pid
    if i >= M:
        return
    
    # Initialize tmp[i] to 0
    tmp_val = 0.0
    
    # Compute tmp[i] = sum(A[i][j] * x[j]) for j in range(N)
    x_offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, N, BLOCK_SIZE):
        current_x_offsets = block_start + x_offsets
        x_mask = current_x_offsets < N
        
        # Load x values
        x_vals = tl.load(x_ptr + current_x_offsets, mask=x_mask, other=0.0)
        
        # Load A[i] values
        a_ptrs = A_ptr + i * A_stride0 + current_x_offsets * A_stride1
        a_vals = tl.load(a_ptrs, mask=x_mask, other=0.0)
        
        # Accumulate tmp[i]
        tmp_val += tl.sum(a_vals * x_vals)
    
    # Store tmp[i]
    tl.store(tmp_ptr + i, tmp_val)
    
    tl.debug_barrier()
    
    # Update y[j] += A[i][j] * tmp[i] for j in range(N)
    y_offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, N, BLOCK_SIZE):
        current_y_offsets = block_start + y_offsets
        y_mask = current_y_offsets < N
        
        # Load A[i] values
        a_ptrs = A_ptr + i * A_stride0 + current_y_offsets * A_stride1
        a_vals = tl.load(a_ptrs, mask=y_mask, other=0.0)
        
        # Load current y values
        y_vals = tl.load(y_ptr + current_y_offsets, mask=y_mask, other=0.0)
        
        # Update y values
        new_y_vals = y_vals + a_vals * tmp_val
        
        # Store updated y values
        tl.store(y_ptr + current_y_offsets, new_y_vals, mask=y_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    grid = (M + 1,)  # +1 for initialization
    
    atax_kernel[grid](
        A, tmp, x, y,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_SIZE
    )