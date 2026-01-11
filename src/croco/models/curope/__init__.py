# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# src/croco/models/curope/__init__.py
try:
    from .curope2d import cuRoPE2D
except Exception as e:
    print(f"Could not import curope2d: {e}")
    # fallback to torch implementation
    pass
