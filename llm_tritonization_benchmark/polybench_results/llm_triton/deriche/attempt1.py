import triton
import triton.language as tl
import torch

@triton.jit
def deriche_kernel(
    imgIn_ptr, imgOut_ptr, y2_ptr, yy1_ptr,
    alpha, H, W,
    BLOCK_SIZE: tl.constexpr
):
    # Compute coefficients
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_neg_2alpha)
    a1 = a5 = k
    a2 = a6 = k * exp_neg_alpha * (alpha - 1.0)
    a3 = a7 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = a8 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)  # 2^(-alpha)
    b2 = -exp_neg_2alpha
    c1 = c2 = 1.0
    
    pid = tl.program_id(0)
    
    # Phase 1: Forward pass for each row i
    if pid < W:
        i = pid
        ym1 = 0.0
        ym2 = 0.0
        xm1 = 0.0
        
        for j in range(H):
            idx = i * H + j
            imgIn_val = tl.load(imgIn_ptr + idx)
            yy1_val = a1 * imgIn_val + a2 * xm1 + b1 * ym1 + b2 * ym2
            tl.store(yy1_ptr + idx, yy1_val)
            
            xm1 = imgIn_val
            ym2 = ym1
            ym1 = yy1_val
    
    tl.debug_barrier()
    
    # Phase 2: Backward pass for each row i
    if pid < W:
        i = pid
        yp1 = 0.0
        yp2 = 0.0
        xp1 = 0.0
        xp2 = 0.0
        
        for j in range(H-1, -1, -1):
            idx = i * H + j
            imgIn_val = tl.load(imgIn_ptr + idx)
            y2_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + idx, y2_val)
            
            xp2 = xp1
            xp1 = imgIn_val
            yp2 = yp1
            yp1 = y2_val
    
    tl.debug_barrier()
    
    # Phase 3: Combine results
    if pid < W:
        i = pid
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = c1 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + idx, imgOut_val)
    
    tl.debug_barrier()
    
    # Phase 4: Forward pass for each column j
    if pid < H:
        j = pid
        tm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        
        for i in range(W):
            idx = i * H + j
            imgOut_val = tl.load(imgOut_ptr + idx)
            yy1_val = a5 * imgOut_val + a6 * tm1 + b1 * ym1 + b2 * ym2
            tl.store(yy1_ptr + idx, yy1_val)
            
            tm1 = imgOut_val
            ym2 = ym1
            ym1 = yy1_val
    
    tl.debug_barrier()
    
    # Phase 5: Backward pass for each column j
    if pid < H:
        j = pid
        tp1 = 0.0
        tp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        
        for i in range(W-1, -1, -1):
            idx = i * H + j
            imgOut_val = tl.load(imgOut_ptr + idx)
            y2_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + idx, y2_val)
            
            tp2 = tp1
            tp1 = imgOut_val
            yp2 = yp1
            yp1 = y2_val
    
    tl.debug_barrier()
    
    # Phase 6: Final combination
    if pid < W:
        i = pid
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = c2 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + idx, imgOut_val)

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 32
    grid = (max(W, H),)
    
    deriche_kernel[grid](
        imgIn, imgOut, y2, yy1,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )