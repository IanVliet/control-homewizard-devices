# For python 3.11 (which is used for the raspberry pi),
# I was not able to use the newest version of the python-homewizard-energy.
# Therefore, to be able to still communicate with a Plug-In battery,
# I modified errors.py from the github:
# https://github.com/homewizard/python-homewizard-energy/tree/main.
# Licensed under the Apache License, Version 2.0

from homewizard_energy.errors import HomeWizardEnergyException


class InvalidUserNameError(HomeWizardEnergyException):
    """Invalid username.

    Raised when username is not valid, too short or too long.
    """


class ResponseError(HomeWizardEnergyException):
    """API responded unexpected."""


class UnauthorizedError(HomeWizardEnergyException):
    """Raised when request is not authorized."""
