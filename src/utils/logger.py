import logging
import sys

# ============================================================================
# FILE OVERVIEW: logger.py
# ============================================================================
# PURPOSE
#   One shared logging setup for the whole app, so every module logs in
#   the same format instead of ad-hoc print() calls scattered around.
#
# GENERAL ALGORITHM
#   1. _configure_root_logger() runs the setup ONCE, no matter how many
#      times get_logger() is called (guarded by _CONFIGURED)
#        1a. attach a handler that writes to stdout
#        1b. give it a formatter: "timestamp | LEVEL | logger name | message"
#        1c. set it on a single root logger named "app", level INFO
#   2. get_logger(name) returns a child logger "app.{name}" - e.g. a module
#      calling get_logger(__name__) gets its own named logger, but they all
#      share the one handler/formatter configured in step 1
#
# WHO CALLS THIS FILE
#   Any module that wants to log something calls get_logger(__name__) once
#   at import time and keeps the returned logger. Currently used in
#   src/api/routes.py's error-handling blocks.
# ============================================================================

_CONFIGURED = False


def _configure_root_logger():
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root_logger = logging.getLogger("app")
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.propagate = False

    _CONFIGURED = True


def get_logger(name):
    _configure_root_logger()
    return logging.getLogger(f"app.{name}")
