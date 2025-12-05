#!/usr/bin/env python3
"""Test all functions in test11 and report results"""

import sys
from pathlib import Path

# Use test11 llm_triton
sys.path.insert(0, str(Path(__file__).parent / 'test11'))
sys.path.insert(0, str(Path(__file__).parent / 'utilities'))
sys.path.insert(0, str(Path(__file__).parent / 'c_reference'))

from tsvc_functions_db import TSVC_FUNCTIONS
from tsvc_all_reference import C_REFERENCE_FUNCS_ALL as C_REFERENCE_FUNCS, get_c_reference_all as get_c_reference

import torch
import inspect

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def test_func(func_name, test_sizes=[100, 1000]):
    c_ref = get_c_reference(func_name)
    if c_ref is None:
        return None, 'no_c_ref'

    try:
        triton_module = __import__(f'llm_triton.{func_name}.attempt1', fromlist=[f'{func_name}_triton'])
        triton_func = getattr(triton_module, f'{func_name}_triton')
    except Exception as e:
        return False, f'import_error:{str(e)[:50]}'

    func_spec = TSVC_FUNCTIONS.get(func_name, {})
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})

    for size in test_sizes:
        try:
            c_tensors = {}
            t_tensors = {}
            for arr_name in sorted(arrays.keys()):
                data = torch.randn(size, device='cuda', dtype=torch.float32)
                c_tensors[arr_name] = data.clone()
                t_tensors[arr_name] = data.clone()

            scalars = {}
            for k, v in scalar_params.items():
                if k != 'iterations':
                    scalars[k] = v

            c_args = build_args(c_ref, c_tensors, scalars)
            c_ref(*c_args)

            t_args = build_args(triton_func, t_tensors, scalars)
            triton_func(*t_args)

            for arr_name in arrays:
                if not torch.allclose(c_tensors[arr_name], t_tensors[arr_name], rtol=1e-3, atol=1e-5):
                    return False, 'numerical'
        except Exception as e:
            return False, f'runtime:{str(e)[:50]}'

    return True, 'passed'

if __name__ == "__main__":
    results = {}
    passed_count = 0
    failed_count = 0

    for i, func_name in enumerate(sorted(TSVC_FUNCTIONS.keys())):
        passed, msg = test_func(func_name)
        results[func_name] = (passed, msg)
        status = "PASS" if passed else "FAIL" if passed == False else "SKIP"
        print(f"[{i+1}/151] {func_name}: {status} ({msg})")
        if passed:
            passed_count += 1
        elif passed == False:
            failed_count += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {len(results) - passed_count - failed_count}")

    print("\nFailed functions:")
    for fn, (p, m) in sorted(results.items()):
        if p == False:
            print(f"  {fn}: {m}")
