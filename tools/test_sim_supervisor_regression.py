#!/usr/bin/env python3
"""Regression checks for simulation and real supervisor path validation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration import real_supervisor, sim_supervisor  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        gmr_python = _touch(root / "gmr" / "python.exe")
        hgpt_python = _touch(root / "hgpt" / "python.exe")
        hgpt_repo = root / "Humanoid-GPT"
        _touch(hgpt_repo / "storage" / "ckpts" / "pns_wo_priv216.onnx")
        _touch(
            hgpt_repo
            / "storage"
            / "ckpts"
            / "G1-Walk"
            / "07140632_G1-Walk_v2.0.0_baseline.onnx"
        )
        dll = _touch(root / "CMVrpn.dll")

        argv = [
            "--workspace-dir",
            str(root),
            "--gmr-python",
            str(gmr_python),
            "--hgpt-python",
            str(hgpt_python),
            "--hgpt-repo",
            str(hgpt_repo),
            "--dll-path",
            str(dll),
        ]
        args = sim_supervisor.build_argument_parser().parse_args(argv)
        paths = sim_supervisor._validate_args(args)
        assert paths.workspace_dir == root.resolve()
        assert paths.gmr_python == gmr_python.resolve()
        assert paths.hgpt_python == hgpt_python.resolve()
        assert paths.hgpt_repo == hgpt_repo.resolve()
        assert paths.dll_path == dll.resolve()
        assert paths.state_store_script.is_file()
        assert paths.gmr_bridge_script.is_file()
        assert paths.hgpt_sim_script.is_file()

        missing_args = sim_supervisor.build_argument_parser().parse_args(
            [*argv[:-1], str(root / "missing.dll")]
        )
        try:
            sim_supervisor._validate_args(missing_args)
        except FileNotFoundError as exc:
            assert "missing.dll" in str(exc)
        except NameError as exc:  # pragma: no cover - explicit regression guard
            raise AssertionError("PipelinePaths was used before construction") from exc
        else:
            raise AssertionError("missing capture DLL was accepted")

        real_argv = [
            "--workspace-dir",
            str(root),
            "--gmr-python",
            str(gmr_python),
            "--hgpt-python",
            str(hgpt_python),
            "--hgpt-repo",
            str(hgpt_repo),
            "--dll-path",
            str(dll),
            "--unitree-interface-address",
            "192.168.123.100",
        ]
        real_args = real_supervisor.build_argument_parser().parse_args(real_argv)
        real_paths = real_supervisor._validate_args(real_args)
        assert real_paths.workspace_dir == root.resolve()
        assert real_paths.real_authorization_script.is_file()
        assert real_paths.play_track_script.is_file()

    print("SUPERVISOR PATH REGRESSION PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
