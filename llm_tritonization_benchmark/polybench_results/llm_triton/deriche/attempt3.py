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
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a1 = k
    a5 = k
    a2 = k * exp_neg_alpha * (alpha - 1.0)
    a6 = k * exp_neg_alpha * (alpha - 1.0)
    a3 = k * exp_neg_alpha * (alpha + 1.0)
    a7 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = -k * exp_neg_2alpha
    a8 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)  # 2^(-alpha)
    b2 = -exp_neg_2alpha
    c1 = 1.0
    c2 = 1.0
    
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

def deriche_phase1(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 32
    grid = (W,)
    
    deriche_kernel[grid](
        imgIn, imgOut, y2, yy1,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )

@triton.jit
def deriche_phase2_kernel(
    imgIn_ptr, y2_ptr,
    alpha, H, W,
    BLOCK_SIZE: tl.constexpr
):
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a3 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
    pid = tl.program_id(0)
    
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

@triton.jit
def deriche_phase3_kernel(
    imgOut_ptr, yy1_ptr, y2_ptr,
    H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid < W:
        i = pid
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = yy1_val + y2_val
            tl.store(imgOut_ptr + idx, imgOut_val)

@triton.jit
def deriche_phase4_kernel(
    imgOut_ptr, yy1_ptr,
    alpha, H, W,
    BLOCK_SIZE: tl.constexpr
):
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a5 = k
    a6 = k * exp_neg_alpha * (alpha - 1.0)
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
    pid = tl.program_id(0)
    
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

@triton.jit
def deriche_phase5_kernel(
    imgOut_ptr, y2_ptr,
    alpha, H, W,
    BLOCK_SIZE: tl.constexpr
):
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a7 = k * exp_neg_alpha * (alpha + 1.0)
    a8 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
    pid = tl.program_id(0)
    
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

@triton.jit
def deriche_phase6_kernel(
    imgOut_ptr, yy1_ptr, y2_ptr,
    H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid < W:
        i = pid
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = yy1_val + y2_val
            tl.store(imgOut_ptr + idx, imgOut_val)

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 32
    
    # Phase 1: Forward pass rows
    grid = (W,)
    deriche_kernel[grid](
        imgIn, imgOut, y2, yy1,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Backward pass rows
    deriche_phase2_kernel[grid](
        imgIn, y2,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 3: Combine row results
    deriche_phase3_kernel[grid](
        imgOut, yy1, y2,
        H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 4: Forward pass columns
    grid = (H,)
    deriche_phase4_kernel[grid](
        imgOut, yy1,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 5: Backward pass columns
    deriche_phase5_kernel[grid](
        imgOut, y2,
        alpha, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 6: Final combination
    grid = (W,)
    deriche_phase6_kernel[grid](
        imgOut, yy1, y2,
        H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )