# For python 3.11 (which is used for the raspberry pi),
# I was not able to use the newest version of the python-homewizard-energy.
# Therefore, to be able to still communicate with a Plug-In battery,
# I copied utils from the github:
# https://github.com/homewizard/python-homewizard-energy/tree/main.
# Licensed under the Apache License, Version 2.0


"""Utilities for Python HomeWizard Energy."""

from functools import lru_cache

from awesomeversion import AwesomeVersion


@lru_cache
def get_awesome_version(version: str) -> AwesomeVersion:
    """Return a cached AwesomeVersion object."""
    if version.lower() == "v1":
        return AwesomeVersion("1.0.0")
    return AwesomeVersion(version)
