"""Read-only local environment discovery for diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
from typing import Dict

from .version import __version__


def _module_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "unavailable"
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as error:  # diagnostic path must survive broken installations
        return "import-error:%s" % type(error).__name__


def collect_config() -> Dict[str, str]:
    return {
        "flashinfer_npu": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "torch_npu": _module_version("torch_npu"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH", "unset"),
        "ascend_opp_path": os.environ.get("ASCEND_OPP_PATH", "unset"),
    }


def format_config() -> str:
    config = collect_config()
    width = max(len(key) for key in config)
    return "\n".join(
        ("%-*s : %s" % (width, key, value)) for key, value in config.items()
    )

