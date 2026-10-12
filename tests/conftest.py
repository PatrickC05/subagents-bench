"""Shared pytest configuration.

Registers two opt-in markers:

* ``live`` -- makes real (paid, fractions of a cent) provider API calls.
  Runs only with ``RUN_LIVE=1``.
* ``docker`` -- needs a running Docker daemon. Runs only with ``RUN_DOCKER=1``.

Without the variable, marked tests are collected and reported as skipped, so
the default ``pytest`` run needs neither network nor Docker.
"""

from __future__ import annotations

import os

import pytest

_OPT_IN = {
    "live": ("RUN_LIVE", "real provider API calls (costs money)"),
    "docker": ("RUN_DOCKER", "needs a running Docker daemon"),
}


def pytest_configure(config: pytest.Config) -> None:
    for marker, (env, what) in _OPT_IN.items():
        config.addinivalue_line("markers", f"{marker}: {what}; runs only with {env}=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, (env, _) in _OPT_IN.items():
        if os.environ.get(env) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{marker} test: set {env}=1 to run")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
