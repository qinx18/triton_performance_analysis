import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr,
                M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Initialize s[j] = 0 for all j
    offsets = tl.arange(0, BLOCK)
    mask = offsets < M
    tl.store(s_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)
    
    # Column accumulator for s[j]
    s_acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Iterate over rows (i dimension)
    for i in range(N):
        # Load row i of matrix A
        col_mask = offsets < M
        a_row = tl.load(A_ptr + i * M + offsets, mask=col_mask)
        
        # Load r[i] (scalar)
        r_i = tl.load(r_ptr + i)
        
        # Load p vector
        p_vec = tl.load(p_ptr + offsets, mask=col_mask)
        
        # Compute q[i] = sum(A[i][j] * p[j]) for all j
        q_i = tl.sum(a_row * p_vec, axis=0)
        tl.store(q_ptr + i, q_i)
        
        # Accumulate s[j] += r[i] * A[i][j]
        s_acc += r_i * a_row
    
    # Store final s values
    tl.store(s_ptr + offsets, s_acc, mask=offsets < M)

def bicg_triton(A, p, q, r, s, M, N):
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    p = p.contiguous()
    q = q.contiguous()
    r = r.contiguous()
    s = s.contiguous()
    
    # Block size for vectorization over M dimension
    BLOCK = triton.next_power_of_2(M)
    
    # Single thread block handles the entire computation
    grid = (1,)
    
    bicg_kernel[grid](
        A, p, q, r, s,
        M=M, N=N, BLOCK=BLOCK
    )