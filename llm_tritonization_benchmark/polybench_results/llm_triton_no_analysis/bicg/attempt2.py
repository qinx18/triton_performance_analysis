import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize s array to zero (first program)
    if pid == 0:
        s_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, M, BLOCK_SIZE):
            current_s_offsets = block_start + s_offsets
            s_mask = current_s_offsets < M
            tl.store(s_ptr + current_s_offsets, 0.0, mask=s_mask)
    
    # Process each row i of N
    row_id = pid - 1
    if row_id >= 0:
        if row_id < N:
            # Initialize q[i] = 0.0
            tl.store(q_ptr + row_id, 0.0)
            
            # Load r[i] once
            r_val = tl.load(r_ptr + row_id)
            
            # Process columns in blocks
            col_offsets = tl.arange(0, BLOCK_SIZE)
            q_acc = 0.0
            
            for block_start in range(0, M, BLOCK_SIZE):
                current_col_offsets = block_start + col_offsets
                col_mask = current_col_offsets < M
                
                # Load A[i][j] values
                a_indices = row_id * M + current_col_offsets
                a_vals = tl.load(A_ptr + a_indices, mask=col_mask, other=0.0)
                
                # Load p[j] values
                p_vals = tl.load(p_ptr + current_col_offsets, mask=col_mask, other=0.0)
                
                # Atomically update s[j] = s[j] + r[i] * A[i][j]
                s_updates = r_val * a_vals
                tl.atomic_add(s_ptr + current_col_offsets, s_updates, mask=col_mask)
                
                # Accumulate q[i] += A[i][j] * p[j]
                q_acc = q_acc + tl.sum(a_vals * p_vals)
            
            # Store final q[i] value
            tl.store(q_ptr + row_id, q_acc)

def bicg_triton(A, p, q, r, s, M, N):
    BLOCK_SIZE = 64
    
    # Launch N+1 programs: one for initialization, N for processing rows
    grid = (N + 1,)
    
    bicg_kernel[grid](
        A, p, q, r, s,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )