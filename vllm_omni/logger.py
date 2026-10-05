from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vllm.logger import configure_logging_from_args, configure_logging_if_needed, init_logger

if TYPE_CHECKING:
    from vllm.config.logging import LoggingConfig

# Logging config applied in this process; handed to spawned children so they
# configure logging the same way (vLLM no longer configures it on import).
_logging_config: LoggingConfig | None = None


def _configure_vllm_omni_root_logger():
    """
    Configure the root logger for vllm_omni to propagate to vllm's root logger.
    """
    vllm_root = logging.getLogger("vllm")
    vllm_omni_root = logging.getLogger("vllm_omni")
    vllm_omni_root.handlers = []

    vllm_omni_root.parent = vllm_root

    vllm_omni_root.propagate = True

    vllm_omni_root.setLevel(logging.NOTSET)


def configure_omni_logging_from_args(args: Any) -> LoggingConfig:
    """Apply the CLI logging arguments in this process, like vLLM's CLI main."""
    global _logging_config
    _logging_config = configure_logging_from_args(args)
    return _logging_config


def configure_omni_logging(config: LoggingConfig | None) -> None:
    """Apply a parent's logging config in a spawned child process.

    ``None`` leaves logging untouched, for callers that start the entry point
    in-process.
    """
    global _logging_config
    if config is None:
        return
    configure_logging_if_needed(config)
    _logging_config = config


def child_logging_config() -> LoggingConfig:
    """Return the logging config to hand to a child process.

    Falls back to the environment-derived defaults when this process was not
    started through the CLI (for example the Python API).
    """
    if _logging_config is not None:
        return _logging_config
    from vllm.config.logging import LoggingConfig

    return LoggingConfig()


_configure_vllm_omni_root_logger()
init_logger(__name__)
