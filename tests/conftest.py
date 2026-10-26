"""Opt-in markers: ``live`` (real Prime and Sail requests, RUN_LIVE=1) and ``docker`` (RUN_DOCKER=1)."""

from __future__ import annotations

import os

import pytest

_OPT_IN = {"live": "RUN_LIVE", "docker": "RUN_DOCKER"}


def pytest_configure(config: pytest.Config) -> None:
    for marker, var in _OPT_IN.items():
        config.addinivalue_line("markers", f"{marker}: runs only with {var}=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, var in _OPT_IN.items():
        if os.environ.get(var) != "1":
            skip = pytest.mark.skip(reason=f"set {var}=1 to run")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)
