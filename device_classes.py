import asyncio
# from homewizard_energy import HomeWizardEnergyV1
from homewizard_energy import HomeWizardEnergy
from contextlib import AsyncExitStack
import logging
import sys


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

    
    async def perform_measurement(self, logger: logging.Logger):
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
    def __init__(self, ip_address: str, device_type, device_name: str, max_power_usage: int | float, priority: int, **kwargs):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self._max_power_usage = max_power_usage
        self.priority = priority
        # the (instantaneous) attributes that change due to each measurement  
        self.inst_state = None
        # whether the device should power on or off
        self.updated_state = False


    @property
    def max_power_usage(self):
        return self._max_power_usage
    
    async def perform_measurement(self, logger: logging.Logger):
        # Get power and current measurement
        measurement = await self.hwe_device.data()
        self.inst_power_usage = measurement.active_power_w
        self.inst_current = measurement.active_current_a

        # get socket state
        device_state = await self.hwe_device.state()
        self.inst_state = device_state.power_on

        # log the power, current and state
        logger.info(f"{self.device_name} power: {self.inst_power_usage}")
        logger.info(f"{self.device_name} current: {self.inst_current}")
        logger.info(f"{self.device_name} power state: {self.inst_state}")

    
    def get_instantaneous_power(self):
        """
        For a socket a positive power indicates the power used by the socket, 
        which can be made free by turning the socket off --> therefore this power should count to available power.
        Since we define available power as negative power the function should return -power
        """
        return -self.inst_power_usage
    

    def should_power_on(self, available_power: int | float):
        return self._max_power_usage <= available_power
    

    async def update_power_state(self, logger: logging.Logger):
        await self.hwe_device.state_set(power_on=self.updated_state)
        logger.info(f"{self.device_name} power state set to: {self.updated_state}")

    

class p1_device(complete_device):
    """
    Class for homewizard p1 meter
    """
    def __init__(self, ip_address, device_type, device_name, **kwargs):
        super().__init__(ip_address, device_type, device_name, **kwargs)

    async def perform_measurement(self, logger: logging.Logger):
        # Get power and current measurement
        measurement = await self.hwe_device.data()
        self.inst_power_usage = measurement.active_power_w
        self.inst_current = measurement.active_current_a

        # log the power and current
        logger.info(f"{self.device_name} power: {self.inst_power_usage}")
        logger.info(f"{self.device_name} current: {self.inst_current}")

    
    def get_instantaneous_power(self):
        """
        For the P1 a negative power means that is the available power,
        therefore we should just return power
        """
        return self.inst_power_usage