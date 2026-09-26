"""Resolve the two isolated Python runtimes and the src-only launch layout."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path

# Resolve the fixed deployment layout locally.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_HGPT_PYTHON = "CHINGMU_GEMINI_NATIVE_HGPT_PYTHON"
_DEVELOPMENT_FALLBACK = (
    PROJECT_ROOT / "tmp" / "windows_hgpt_runtime_312" / "python.exe"
)

# Frozen legacy-chain names (moved here from src/domain/process_spec.py:
# domain declares the abstract layout, adapters own the concrete names).
HGPT_POLICY_NAME = "pns_wo_priv216.onnx"
G1_WALK_NAME = "07140632_G1-Walk_v2.0.0_baseline.onnx"


class RuntimeAdapter:
    """Read-only resolution of interpreters, DLL, and repository paths."""

    @property
    def workspace_dir(self) -> Path:
        """Workspace used as the child-process working directory."""
        return PROJECT_ROOT

    @property
    def package_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def gmr_python(self) -> Path:
        """The interpreter running the control service (runtime-gmr)."""
        return Path(sys.executable).resolve()

    @property
    def hgpt_python(self) -> Path:
        candidate = Path(
            os.environ.get(
                ENV_HGPT_PYTHON,
                str(PROJECT_ROOT / "runtime-hgpt" / "python.exe"),
            )
        ).resolve()
        if candidate.is_file():
            return candidate
        development = _DEVELOPMENT_FALLBACK.resolve()
        if development.is_file():
            return development
        return candidate

    @property
    def hgpt_repo(self) -> Path:
        return PROJECT_ROOT

    @property
    def dll_path(self) -> Path:
        return (
            Path(sys.executable).resolve().parent / "ChingmuDLL" / "CMVrpn.dll"
        )

    @property
    def lowstate_report(self) -> Path:
        return PROJECT_ROOT / "storage" / "reports" / "unitree_lowstate_windows_report.json"

    def require_hgpt_python(self) -> Path:
        path = self.hgpt_python
        if not path.is_file():
            raise FileNotFoundError(f"Humanoid-GPT runtime is missing: {path}")
        return path


@dataclass(frozen=True)
class PipelinePaths:
    """Fixed absolute paths used by every supervisor launch.

    Moved here from src/domain/process_spec.py (wrap-up audit): the concrete
    deployment layout is an adapter concern; domain keeps only the abstract
    ``PipelineLayout`` contract.  Satisfies that protocol structurally.
    """

    workspace_dir: Path
    gmr_python: Path
    hgpt_python: Path
    hgpt_repo: Path
    dll_path: Path
    play_track_module: str = "src.motion.play_track"

    @property
    def package_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def sim_pipeline_script(self) -> Path:
        return PROJECT_ROOT / "src" / "orchestration" / "sim_supervisor.py"

    @property
    def real_pipeline_script(self) -> Path:
        return PROJECT_ROOT / "src" / "orchestration" / "real_supervisor.py"

    @property
    def preview_pipeline_script(self) -> Path:
        return PROJECT_ROOT / "src" / "orchestration" / "preview_supervisor.py"

    @property
    def state_store_script(self) -> Path:
        return PROJECT_ROOT / "src" / "adapters" / "state_store.py"

    @property
    def gmr_bridge_script(self) -> Path:
        return PROJECT_ROOT / "src" / "motion" / "gmr_bridge.py"

    @property
    def chingmu_sender_script(self) -> Path:
        return PROJECT_ROOT / "src" / "adapters" / "chingmu.py"

    @property
    def replay_sender_script(self) -> Path:
        return PROJECT_ROOT / "src" / "motion" / "replay.py"

    @property
    def hgpt_sim_script(self) -> Path:
        return PROJECT_ROOT / "src" / "motion" / "live_sim.py"

    @property
    def gmr_preview_exact_script(self) -> Path:
        return PROJECT_ROOT / "src" / "motion" / "gmr_preview.py"

    @property
    def real_authorization_script(self) -> Path:
        return PROJECT_ROOT / "src" / "application" / "authorization.py"

    @property
    def play_track_script(self) -> Path:
        return PROJECT_ROOT / "src" / "motion" / "play_track.py"

    @property
    def windows_unitree_script(self) -> Path:
        return PROJECT_ROOT / "src" / "adapters" / "unitree.py"


def default_pipeline_paths(
    *,
    workspace_dir: Path | None = None,
    gmr_python: Path | None = None,
    hgpt_python: Path | None = None,
    dll_path: Path | None = None,
) -> PipelinePaths:
    """Resolve the src-only deployment paths used by all supervisors."""
    runtime = RuntimeAdapter()
    return PipelinePaths(
        workspace_dir=(workspace_dir or runtime.workspace_dir).expanduser().resolve(),
        gmr_python=(gmr_python or runtime.gmr_python).expanduser().resolve(),
        hgpt_python=(hgpt_python or runtime.hgpt_python).expanduser().resolve(),
        hgpt_repo=runtime.hgpt_repo,
        dll_path=(dll_path or runtime.dll_path).expanduser().resolve(),
    )
