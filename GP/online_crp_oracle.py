"""Load the Mijn_RL OCRP semantic oracle without importing its Torch package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_NAME = "GP._online_crp_truth"
_SOURCE = (
    Path(__file__).resolve().parents[1] / "Mijn_RL" / "scrp" / "online_crp.py"
)

if _MODULE_NAME in sys.modules:
    _oracle = sys.modules[_MODULE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load OCRP semantic oracle from {_SOURCE}")
    _oracle = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _oracle
    _spec.loader.exec_module(_oracle)

LevelingBaseline = _oracle.LevelingBaseline
OnlineCRPEpisode = _oracle.OnlineCRPEpisode
OnlineCRPEnvironment = _oracle.OnlineCRPEnvironment
OnlineCRPPublicState = _oracle.OnlineCRPPublicState
load_episode_manifest = _oracle.load_episode_manifest


__all__ = [
    "LevelingBaseline",
    "OnlineCRPEpisode",
    "OnlineCRPEnvironment",
    "OnlineCRPPublicState",
    "load_episode_manifest",
]
