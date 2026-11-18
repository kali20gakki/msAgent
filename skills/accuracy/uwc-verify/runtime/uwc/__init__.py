"""UWC runtime: collectors and offline comparators for veRL weight consistency.

This package is copied into ``<verl>/verl/utils/uwc`` by ``uwc-verify prepare``
so that veRL can import it as ``verl.utils.uwc``.  It is intentionally
dependency-light: importing the package must not import torch, verl or vLLM.
"""

__version__ = "1.0.0"
RUNTIME_MARKER = "uwc-skill-runtime"
