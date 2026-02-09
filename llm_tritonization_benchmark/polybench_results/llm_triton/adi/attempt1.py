import triton
import triton.language as tl
import torch

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, N, TSTEPS):
    # Compute coefficients
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)
    
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    # Time loop
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        for i in range(1, N - 1):
            # Set boundary values
            v_idx = 0 * N + i
            tl.store(v_ptr + v_idx, 1.0)
            
            p_idx = i * N + 0
            tl.store(p_ptr + p_idx, 0.0)
            
            q_idx = i * N + 0
            v_val = tl.load(v_ptr + v_idx)
            tl.store(q_ptr + q_idx, v_val)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_curr_idx = i * N + j
                p_prev_idx = i * N + (j - 1)
                q_curr_idx = i * N + j
                q_prev_idx = i * N + (j - 1)
                
                p_prev = tl.load(p_ptr + p_prev_idx)
                q_prev = tl.load(q_ptr + q_prev_idx)
                
                u_idx1 = j * N + (i - 1)
                u_idx2 = j * N + i
                u_idx3 = j * N + (i + 1)
                
                u_val1 = tl.load(u_ptr + u_idx1)
                u_val2 = tl.load(u_ptr + u_idx2)
                u_val3 = tl.load(u_ptr + u_idx3)
                
                p_new = -c / (a * p_prev + b)
                q_new = (-d * u_val1 + (1.0 + 2.0 * d) * u_val2 - f * u_val3 - a * q_prev) / (a * p_prev + b)
                
                tl.store(p_ptr + p_curr_idx, p_new)
                tl.store(q_ptr + q_curr_idx, q_new)
            
            # Set boundary
            v_end_idx = (N - 1) * N + i
            tl.store(v_ptr + v_end_idx, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                v_curr_idx = j * N + i
                v_next_idx = (j + 1) * N + i
                p_idx = i * N + j
                q_idx = i * N + j
                
                v_next = tl.load(v_ptr + v_next_idx)
                p_val = tl.load(p_ptr + p_idx)
                q_val = tl.load(q_ptr + q_idx)
                
                v_new = p_val * v_next + q_val
                tl.store(v_ptr + v_curr_idx, v_new)
        
        # Row Sweep
        for i in range(1, N - 1):
            # Set boundary values
            u_idx = i * N + 0
            tl.store(u_ptr + u_idx, 1.0)
            
            p_idx = i * N + 0
            tl.store(p_ptr + p_idx, 0.0)
            
            q_idx = i * N + 0
            u_val = tl.load(u_ptr + u_idx)
            tl.store(q_ptr + q_idx, u_val)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_curr_idx = i * N + j
                p_prev_idx = i * N + (j - 1)
                q_curr_idx = i * N + j
                q_prev_idx = i * N + (j - 1)
                
                p_prev = tl.load(p_ptr + p_prev_idx)
                q_prev = tl.load(q_ptr + q_prev_idx)
                
                v_idx1 = (i - 1) * N + j
                v_idx2 = i * N + j
                v_idx3 = (i + 1) * N + j
                
                v_val1 = tl.load(v_ptr + v_idx1)
                v_val2 = tl.load(v_ptr + v_idx2)
                v_val3 = tl.load(v_ptr + v_idx3)
                
                p_new = -f / (d * p_prev + e)
                q_new = (-a * v_val1 + (1.0 + 2.0 * a) * v_val2 - c * v_val3 - d * q_prev) / (d * p_prev + e)
                
                tl.store(p_ptr + p_curr_idx, p_new)
                tl.store(q_ptr + q_curr_idx, q_new)
            
            # Set boundary
            u_end_idx = i * N + (N - 1)
            tl.store(u_ptr + u_end_idx, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                u_curr_idx = i * N + j
                u_next_idx = i * N + (j + 1)
                p_idx = i * N + j
                q_idx = i * N + j
                
                u_next = tl.load(u_ptr + u_next_idx)
                p_val = tl.load(p_ptr + p_idx)
                q_val = tl.load(q_ptr + q_idx)
                
                u_new = p_val * u_next + q_val
                tl.store(u_ptr + u_curr_idx, u_new)

def adi_triton(p, q, u, v, N, TSTEPS):
    adi_kernel[(1,)](p, q, u, v, N, TSTEPS)