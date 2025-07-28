import asyncio

# from homewizard_energy import HomeWizardEnergyV1
from homewizard_energy import HomeWizardEnergy
from .hwe_v2_wrapper.init_wrapper import HomeWizardEnergyV2
from contextlib import AsyncExitStack
import logging
import sys
from dataclasses import dataclass


class complete_device:
    """
    Class for all properties of the device
    """

    def __init__(self, ip_address, device_type, device_name, **kwargs):
        self._ip_address = ip_address
        self._device_type = device_type
        self._device_name = device_name
        self._hwe_device = None

        # the (instantaneous) attributes that change due to each measurement
        self.inst_power_usage = None
        self.inst_current = None

    @property
    def ip_address(self):
        return self._ip_address

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_name(self):
        return self._device_name

    @property
    def hwe_device(self):
        return self._hwe_device

    @hwe_device.setter
    def hwe_device(self, device: HomeWizardEnergy):
        self._hwe_device = device

    def get_HWE_class(self):
        return HomeWizardEnergy(host=self.ip_address)

    async def perform_measurement(self, logger: logging.Logger):
        if self.hwe_device is not None:
            # Get device information, like firmware version
            hwe_device_info = await self.hwe_device.device()
            logger.info(hwe_device_info)

            # Get measurement --> power and current
            measurement = await self.hwe_device.data()
            self.inst_power_usage = measurement.active_power_w
            self.inst_current = measurement.active_current_a

            # log power and current
            logger.info(f"{self.device_name} power: {self.inst_power_usage}")
            logger.info(f"{self.device_name} current: {self.inst_current}")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")

    def get_instantaneous_power(self):
        """
        For a general device, we assume that this power can not be freed,
        therefore we should just return power
        """
        return self.inst_power_usage


class socket_device(complete_device):
    """
    Class for homewizard energy socket
    """

    def __init__(
        self,
        ip_address: str,
        device_type,
        device_name: str,
        max_power_usage: int | float,
        energy_capacity: int | float,
        priority: int,
        daily_need: bool,
        **kwargs,
    ):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self._max_power_usage = max_power_usage
        self.energy_capacity = energy_capacity
        self.priority = priority
        self.daily_need = daily_need
        # the (instantaneous) attributes that change due to each measurement
        self.inst_state = None
        # whether the device should power on or off
        self.updated_state = False
        self.energy_stored = 0.0

    @property
    def max_power_usage(self):
        return self._max_power_usage

    async def perform_measurement(self, logger: logging.Logger):
        if self.hwe_device is not None:
            # Get power and current measurement
            measurement = await self.hwe_device.data()
            self.inst_power_usage = measurement.active_power_w
            self.inst_current = measurement.active_current_a

            # get socket state
            device_state = await self.hwe_device.state()
            if device_state is not None:
                self.inst_state = device_state.power_on
            else:
                logger.warning(f"{self.device_name}'s device state is None")

            # log the power, current and state
            logger.info(f"{self.device_name} power: {self.inst_power_usage}")
            logger.info(f"{self.device_name} current: {self.inst_current}")
            logger.info(f"{self.device_name} power state: {self.inst_state}")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")

    def get_instantaneous_power(self):
        """
        For a socket a positive power indicates the power used by the socket,
        which can be made free by turning the socket off --> therefore this power should count to available power.
        Since we define available power as negative power the function should return -power
        """
        if self.inst_power_usage is None:
            return None
        else:
            return -self.inst_power_usage

    def should_power_on(self, available_power: int | float):
        return self._max_power_usage <= available_power

    async def update_power_state(self, logger: logging.Logger):
        if self.hwe_device is not None:
            await self.hwe_device.state_set(power_on=self.updated_state)
            logger.info(f"{self.device_name} power state set to: {self.updated_state}")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")


class p1_device(complete_device):
    """
    Class for homewizard p1 meter
    """

    def __init__(self, ip_address, device_type, device_name, **kwargs):
        super().__init__(ip_address, device_type, device_name, **kwargs)

    async def perform_measurement(self, logger: logging.Logger):
        if self.hwe_device is not None:
            # Get power and current measurement
            measurement = await self.hwe_device.data()
            self.inst_power_usage = measurement.active_power_w
            self.inst_current = measurement.active_current_a

            # log the power and current
            logger.info(f"{self.device_name} power: {self.inst_power_usage}")
            logger.info(f"{self.device_name} current: {self.inst_current}")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")

    def get_instantaneous_power(self):
        """
        For the P1 a negative power means that is the available power,
        therefore we should just return power
        """
        return self.inst_power_usage


@dataclass
class UserInfo:
    name: str
    token: str


class Battery(complete_device):
    """
    Homewizard Battery
    """

    def __init__(
        self,
        ip_address: str,
        device_type: str,
        device_name: str,
        max_power_usage: int | float,
        energy_capacity: int | float,
        user_info: dict[str, str],
        **kwargs,
    ):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self.max_power_usage = max_power_usage
        self.energy_capacity = energy_capacity
        self._user_info = UserInfo(**user_info)
        self._token = self._user_info.token

        self.state_of_charge_pct = None
        self.stored_energy = None

    @property
    def hwe_device(self):
        return self._hwe_device

    @hwe_device.setter
    def hwe_device(self, device: HomeWizardEnergyV2):
        self._hwe_device = device

    def get_HWE_class(self):
        return HomeWizardEnergyV2(host=self.ip_address, token=self._token)

    async def perform_measurement(self, logger: logging.Logger):
        if self.hwe_device is not None:
            measurement = await self.hwe_device.measurement()
            self.inst_power_usage = measurement.power_w
            self.state_of_charge_pct = measurement.state_of_charge_pct
            if self.state_of_charge_pct:
                self.stored_energy = (
                    self.energy_capacity * self.state_of_charge_pct / 100
                )
            # log the power and current
            logger.info(f"{self.device_name} power: {self.inst_power_usage}")
            logger.info(f"{self.device_name} percentage: {self.state_of_charge_pct} %")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")
