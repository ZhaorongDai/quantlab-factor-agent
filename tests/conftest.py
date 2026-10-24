"""Test session setup.

macOS only: xgboost's wheel links Homebrew's libomp and torch bundles its
own, so a process loading both segfaults (Acceptance trains xgb after
quantlab has loaded torch). Single-threaded OpenMP is the setting quantlab's
own test suite found safe; it must be set before torch loads, so it stays
ahead of every other import. Linux is untouched.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
