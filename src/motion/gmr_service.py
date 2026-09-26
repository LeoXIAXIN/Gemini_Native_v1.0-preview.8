"""GMRService: semantic facade over the canonical GMR bridge primitives.

RUNTIME: gmr (3.10) — imports ``general_motion_retargeting`` lazily.

Delegation points:
  * human-frame construction -> ``src.motion.gmr_exact.make_exact_gmr_frame``
  * qpos->mimic conversion    -> ``src.motion.gmr_twist.gmr_qpos_to_twist_mimic``
  * retargeting               -> ``general_motion_retargeting.GeneralMotionRetargeting``
    (constructed lazily at the adapter boundary)

The reference-packet fields include session, sequence, freshness and DoF hash.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.adapters.gmr import GMRAdapter
from src.adapters.gmr_headless import load_general_motion_retargeting
from src.domain.models import G1Reference, MocapFrame

ROBOT_MODEL = "unitree_g1_29dof"


class GMRService:
    """GMR retarget/reference facade; all math stays in the legacy modules."""

    def __init__(self, adapter: GMRAdapter | None = None) -> None:
        self._adapter = adapter or GMRAdapter()

    def make_retargeter(self, human_height: float, verbose: bool = False) -> Any:
        """Construct the GMR retargeter exactly like the legacy preflight."""
        return load_general_motion_retargeting()(
            src_human="bvh_nokov",
            tgt_robot="unitree_g1",
            actual_human_height=float(human_height),
            verbose=verbose,
        )

    def to_human_frame(self, frame: MocapFrame) -> dict[str, list[np.ndarray]]:
        """CHINGMU live frame -> GMR human frame (legacy exact math)."""
        return self._adapter.make_exact_gmr_frame(
            frame.positions, frame.rotations
        )

    def retarget(self, retargeter: Any, frame: MocapFrame) -> np.ndarray:
        """Run the legacy retargeter on one decoded frame; returns qpos (36,)."""
        return np.asarray(
            retargeter.retarget(self.to_human_frame(frame)), dtype=float
        ).copy()

    def to_mimic(
        self, qpos: np.ndarray, timestamp: float, velocity_estimator: Any
    ) -> np.ndarray:
        """GMR qpos (36,) -> TWIST mimic target (33,) (legacy math)."""
        return self._adapter.gmr_qpos_to_twist_mimic(
            qpos, timestamp, velocity_estimator
        )

    def dof_order_hash(self) -> str:
        """The pinned 29-DoF order hash (same formula as the legacy bridge)."""
        names = self._adapter.gmr_29_dof_names
        return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()

    def build_reference(
        self,
        qpos: np.ndarray,
        *,
        session_id: str,
        sequence: int,
        frame_id: int,
        generated_monotonic: float,
        valid_until_monotonic: float,
        mimic: np.ndarray | None = None,
    ) -> G1Reference:
        """Wrap a retargeted qpos into the frozen ``action_qpos_g1_packet`` model."""
        return G1Reference(
            session_id=session_id,
            sequence=int(sequence),
            frame_id=int(frame_id),
            generated_monotonic=float(generated_monotonic),
            valid_until_monotonic=float(valid_until_monotonic),
            robot_model=ROBOT_MODEL,
            dof_order_hash=self.dof_order_hash(),
            qpos=np.asarray(qpos, dtype=float),
            mimic=mimic,
        )
