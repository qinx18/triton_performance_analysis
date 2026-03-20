import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr
):
    # Sequential execution of time steps
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t] for j = 0 to NY-1
        fict_val = tl.load(_fict_ptr + t)
        for j in range(NY):
            ey_idx = 0 * NY + j
            tl.store(ey_ptr + ey_idx, fict_val)
        
        # Step 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j])
        for i in range(1, NX):
            for j in range(NY):
                ey_idx = i * NY + j
                hz_idx = i * NY + j
                hz_prev_idx = (i - 1) * NY + j
                
                ey_val = tl.load(ey_ptr + ey_idx)
                hz_val = tl.load(hz_ptr + hz_idx)
                hz_prev_val = tl.load(hz_ptr + hz_prev_idx)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_idx, new_ey)
        
        # Step 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1])
        for i in range(NX):
            for j in range(1, NY):
                ex_idx = i * NY + j
                hz_idx = i * NY + j
                hz_prev_j_idx = i * NY + (j - 1)
                
                ex_val = tl.load(ex_ptr + ex_idx)
                hz_val = tl.load(hz_ptr + hz_idx)
                hz_prev_j_val = tl.load(hz_ptr + hz_prev_j_idx)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev_j_val)
                tl.store(ex_ptr + ex_idx, new_ex)
        
        # Step 4: hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz_idx = i * NY + j
                ex_idx = i * NY + j
                ex_next_j_idx = i * NY + (j + 1)
                ey_idx = i * NY + j
                ey_next_i_idx = (i + 1) * NY + j
                
                hz_val = tl.load(hz_ptr + hz_idx)
                ex_val = tl.load(ex_ptr + ex_idx)
                ex_next_j = tl.load(ex_ptr + ex_next_j_idx)
                ey_val = tl.load(ey_ptr + ey_idx)
                ey_next_i = tl.load(ey_ptr + ey_next_i_idx)
                
                new_hz = hz_val - 0.7 * (ex_next_j - ex_val + ey_next_i - ey_val)
                tl.store(hz_ptr + hz_idx, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    grid = (1,)
    
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX, NY, TMAX
    )