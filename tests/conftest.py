"""Shared pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that hit the real Wood Wide API (deselect with -m 'not integration')"
    )
