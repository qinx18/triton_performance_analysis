#!/usr/bin/env python3
"""
Correctness Test for s221
Tests: Original vs Corrected implementations against sequential C execution
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s221_baseline import s221_pytorch as s221_baseline
    from baselines.s221_baseline_correct import s221_pytorch as s221_correct
    from llm_triton.s221_triton_llm import s221_triton as s221_triton_llm
    from llm_triton.s221_triton_correct import s221_triton as s221_triton_correct
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def s221_sequential_c(a, b, c, d):
    """
    True sequential C execution (ground truth):
    for (int i = 1; i < n; i++) {
        a[i] += c[i] * d[i];
        b[i] = b[i-1] + a[i] + d[i];
    }
    """
    a = a.clone()
    b = b.clone()
    c = c.clone()
    d = d.clone()

    n = a.shape[0]
    for i in range(1, n):
        a[i] += c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]

    return a, b

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]

    print("="*80)
    print(f"s221 Correctness Testing: Original vs Corrected")
    print("="*80)

    # First, small test to show differences
    print("\n--- Small Test (N=10) to Show Differences ---\n")
    N = 10
    torch.manual_seed(42)
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(N, device='cuda', dtype=torch.float32)
    d = torch.randn(N, device='cuda', dtype=torch.float32)

    a_seq, b_seq = s221_sequential_c(a.clone(), b.clone(), c.clone(), d.clone())
    a_baseline, b_baseline = s221_baseline(a.clone(), b.clone(), c.clone(), d.clone())
    a_correct, b_correct = s221_correct(a.clone(), b.clone(), c.clone(), d.clone())
    a_triton_llm, b_triton_llm = s221_triton_llm(a.clone(), b.clone(), c.clone(), d.clone())
    a_triton_correct, b_triton_correct = s221_triton_correct(a.clone(), b.clone(), c.clone(), d.clone())

    print("Array a (first 5 elements):")
    print(f"  Sequential C:       {a_seq[:5].cpu().numpy()}")
    print(f"  Original Baseline:  {a_baseline[:5].cpu().numpy()}")
    print(f"  Corrected Baseline: {a_correct[:5].cpu().numpy()}")
    print(f"  Original Triton:    {a_triton_llm[:5].cpu().numpy()}")
    print(f"  Corrected Triton:   {a_triton_correct[:5].cpu().numpy()}")

    print("\nArray b (first 5 elements):")
    print(f"  Sequential C:       {b_seq[:5].cpu().numpy()}")
    print(f"  Original Baseline:  {b_baseline[:5].cpu().numpy()}")
    print(f"  Corrected Baseline: {b_correct[:5].cpu().numpy()}")
    print(f"  Original Triton:    {b_triton_llm[:5].cpu().numpy()}")
    print(f"  Corrected Triton:   {b_triton_correct[:5].cpu().numpy()}")

    # Check against sequential C
    def check_error(name, a_result, b_result):
        error_a = torch.max(torch.abs(a_result - a_seq)).item()
        error_b = torch.max(torch.abs(b_result - b_seq)).item()
        max_err = max(error_a, error_b)
        status = "✓" if max_err < 1e-4 else "❌"
        print(f"  {status} {name}: a_err={error_a:.2e}, b_err={error_b:.2e}")
        return max_err < 1e-4

    print("\nError vs Sequential C:")
    baseline_ok = check_error("Original Baseline ", a_baseline, b_baseline)
    correct_ok = check_error("Corrected Baseline", a_correct, b_correct)
    triton_llm_ok = check_error("Original Triton   ", a_triton_llm, b_triton_llm)
    triton_correct_ok = check_error("Corrected Triton  ", a_triton_correct, b_triton_correct)

    # Now test larger sizes with corrected version
    print("\n" + "="*80)
    print("Testing Corrected Versions on Larger Sizes:")
    print("="*80)

    all_passed = True
    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            torch.manual_seed(42)
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            c = torch.randn(N, device='cuda', dtype=torch.float32)
            d = torch.randn(N, device='cuda', dtype=torch.float32)

            # Run sequential C (ground truth)
            a_seq, b_seq = s221_sequential_c(a.clone(), b.clone(), c.clone(), d.clone())

            # Run corrected versions
            a_correct, b_correct = s221_correct(a.clone(), b.clone(), c.clone(), d.clone())
            a_triton, b_triton = s221_triton_correct(a.clone(), b.clone(), c.clone(), d.clone())

            # Compare against sequential C
            error_baseline_a = torch.max(torch.abs(a_correct - a_seq)).item()
            error_baseline_b = torch.max(torch.abs(b_correct - b_seq)).item()
            error_triton_a = torch.max(torch.abs(a_triton - a_seq)).item()
            error_triton_b = torch.max(torch.abs(b_triton - b_seq)).item()

            max_error = max(error_baseline_a, error_baseline_b, error_triton_a, error_triton_b)

            if max_error < 1e-3:
                print(f"✓ PASS  (baseline_err={max(error_baseline_a, error_baseline_b):.2e}, triton_err={max(error_triton_a, error_triton_b):.2e})")
            else:
                print(f"✗ FAIL  (max_error={max_error:.2e})")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False

    print("="*80)
    print("\nSummary:")
    print("="*80)
    if baseline_ok and triton_llm_ok:
        print("✓ Original implementations are CORRECT (use sequential execution)")
        print("  - But they are slow (sequential in-kernel loop)")
    else:
        print("❌ Original implementations have bugs")

    if correct_ok and triton_correct_ok:
        print("✓ Corrected implementations are CORRECT")
        print("  - AND much faster (parallel a[] + efficient cumsum for b[])")
    else:
        print("❌ Corrected implementations have bugs")

    if all_passed:
        print("\n✅ All corrected version tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
