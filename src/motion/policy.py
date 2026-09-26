"""M5 PolicyService: semantic facade over the legacy ONNX policy loader.

RUNTIME: hgpt (3.12).  Importing this module is safe anywhere; loading a
policy raises ``DomainError`` when onnxruntime is unavailable (e.g. under the
gmr runtime).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.adapters.onnx import OnnxRuntimeAdapter


class PolicyService:
    """Loads and runs the legacy ONNX policy without re-implementing it."""

    def __init__(self, adapter: OnnxRuntimeAdapter | None = None) -> None:
        self._adapter = adapter or OnnxRuntimeAdapter()

    def load_policy(
        self,
        load_path: str,
        *,
        device: str = "cpu",
        policy_type: str = "mlp",
        use_trt: bool = False,
        strict_trt: bool = False,
    ) -> Any:
        args = self._adapter.make_args(
            policy_type=policy_type, device=device, load_path=load_path
        )
        return self._adapter.get_policy_onnx(
            args, use_trt=use_trt, strict_trt=strict_trt
        )

    def infer(self, policy: Any, observation: np.ndarray) -> np.ndarray:
        """obs -> continuous_actions (legacy ``policy.infer``)."""
        return np.asarray(policy.infer(observation))
