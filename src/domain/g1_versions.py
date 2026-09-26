"""Internal G1 model assets, independent of the official SDK wire protocol.

The profile selects MuJoCo assets and the corresponding policy configuration.
It never selects a DDS type or predicts ``mode_machine``. Real commands copy
that field from validated live LowState feedback, as in Unitree's example.
"""

from __future__ import annotations

from typing import Final, Mapping


DEFAULT_MODEL_PROFILE: Final[str] = "unitree_g1_5010"
SUPPORTED_MODEL_PROFILES: Final[tuple[str, ...]] = (
    "unitree_g1_4010",
    "unitree_g1_5010",
)


def require_model_profile(value: object) -> str:
    """Validate a packaged model asset directory, without inferring hardware."""

    profile = str(value).strip()
    if profile not in SUPPORTED_MODEL_PROFILES:
        raise ValueError(
            "model_profile must be " + " or ".join(SUPPORTED_MODEL_PROFILES)
        )
    return profile


def model_profile_from_config(config: Mapping[str, object]) -> str:
    """Read an internal profile, preserving the model in an older saved file.

    ``g1_version`` is accepted only as a migration input. The explicit new
    profile wins if both are present; no mode value participates in migration.
    """

    if "model_profile" in config:
        return require_model_profile(config["model_profile"])
    if "g1_version" in config:
        return require_model_profile(f"unitree_g1_{str(config['g1_version']).strip()}")
    return DEFAULT_MODEL_PROFILE
