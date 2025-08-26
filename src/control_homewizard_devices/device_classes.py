from homewizard_energy import HomeWizardEnergy
from control_homewizard_devices.hwe_v2_wrapper.init_wrapper import HomeWizardEnergyV2
import logging
from dataclasses import dataclass

from control_homewizard_devices.constants import (
    DELTA_T,
    IS_FULL_POWER_RATIO,
    PERIODIC_SLEEP_DURATION,
)


class CompleteDevice:
    """
    Class for all properties of the device
    """

    def __init__(self, ip_address: str, device_type: str, device_name: str, **kwargs):
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
        """
        Perform a measurement for the device.
        This method should be overridden in subclasses to implement specific measurement logic.

        For example, certain devices would also update the energy stored based on the instantaneous power usage or state of charge.
        """
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


class P1Device(CompleteDevice):
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


class SocketDevice(CompleteDevice):
    """
    Class for homewizard energy socket
    """

    def __init__(
        self,
        ip_address: str,
        device_type,
        device_name: str,
        max_power_usage: float,
        energy_capacity: float,
        priority: int,
        daily_need: bool,
        delta_t: float = DELTA_T,
        **kwargs,
    ):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self._max_power_usage = max_power_usage
        self.energy_capacity = energy_capacity
        self.policy = SocketDeviceSchedulePolicy(self, delta_t)
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
        """
        Perform a measurement for the socket device.
        This method gets the instantaneous power usage, current, and state of the socket.
        It also updates the energy stored based on the instantaneous power usage.
        """
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
            logger.info(f"{self.device_name} power: {self.inst_power_usage} W")
            logger.debug(f"{self.device_name} current: {self.inst_current}")
            logger.info(f"{self.device_name} power state: {self.inst_state}")
            # Update the energy stored based on the instantaneous power usage
            self.update_energy_stored(logger)
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")

    def update_energy_stored(self, logger: logging.Logger):
        """
        Update the energy stored in the socket based on the current power usage.
        """
        if self.inst_power_usage is not None:
            if (
                self.inst_state
                and self.inst_power_usage < IS_FULL_POWER_RATIO * self.max_power_usage
            ):
                # If the socket is on and the power usage is below the full power ratio,
                # we consider it fully charged
                self.energy_stored = self.energy_capacity
            elif (
                self.inst_state
                and self.inst_power_usage >= IS_FULL_POWER_RATIO * self.max_power_usage
                and not self.updated_state
            ):
                # If the socket is and the power usage is above the expected power usage of a full device,
                # and the state is not set to the state determined by the scheduler,
                # we assume that a user has turned a socket on and we have detected that the device is not (or no longer) fully charged
                self.energy_stored = 0.0
            else:
                # Note: Updates with the constant sleep time for simplicity and due to the energy stored of the sockets (measurement.total_power_import_kwh) only having W precision.
                self.energy_stored += (
                    self.inst_power_usage * PERIODIC_SLEEP_DURATION / 3600
                )
            logger.info(f"{self.device_name} energy stored: {self.energy_stored} Wh")

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


@dataclass
class UserInfo:
    name: str
    token: str


# TODO: Remove priority and daily need from Battery class, as they are not used.
class Battery(CompleteDevice):
    """
    Homewizard Battery
    """

    def __init__(
        self,
        ip_address: str,
        device_type: str,
        device_name: str,
        max_power_usage: float,
        energy_capacity: float,
        user_info: dict[str, str],
        **kwargs,
    ):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self.max_power_usage = max_power_usage
        self.energy_capacity = energy_capacity
        # Batteries do not have a daily need (they never draw from the grid)
        self._user_info = UserInfo(**user_info)
        self._token = self._user_info.token

        self.policy = BatterySchedulePolicy(self)
        self.state_of_charge_pct = None
        self.energy_stored = 0.0

    @property
    def hwe_device(self):
        return self._hwe_device

    @hwe_device.setter
    def hwe_device(self, device: HomeWizardEnergyV2):
        self._hwe_device = device

    def get_HWE_class(self):
        return HomeWizardEnergyV2(host=self.ip_address, token=self._token)

    async def perform_measurement(self, logger: logging.Logger):
        """
        Perform a measurement for the battery device.
        This method gets the instantaneous power usage and state of charge percentage.
        It also updates the energy stored based on the state of charge.
        """
        if self.hwe_device is not None:
            measurement = await self.hwe_device.measurement()
            self.inst_power_usage = measurement.power_w
            self.state_of_charge_pct = measurement.state_of_charge_pct
            # log the power and current
            logger.info(f"{self.device_name} power: {self.inst_power_usage} W")
            logger.info(f"{self.device_name} percentage: {self.state_of_charge_pct} %")
            # Update the energy stored based on the state of charge
            self.update_energy_stored(logger)
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")

    def update_energy_stored(self, logger: logging.Logger):
        """
        Update the energy stored in the battery based on the current state of charge.
        """
        if self.state_of_charge_pct is not None:
            self.energy_stored = self.energy_capacity * self.state_of_charge_pct / 100
            logger.info(f"{self.device_name} energy stored: {self.energy_stored} Wh")


class DeviceSchedulePolicy:
    """
    Data class for fields determining the properties of the variables used when scheduling devices based on power forecasts.
    """

    def __init__(self, device: CompleteDevice) -> None:
        self.device = device
        self.energy_stored_upper: float


class BatterySchedulePolicy(DeviceSchedulePolicy):
    """
    Class for fields determining the properties of the variables used when scheduling batteries based on power forecasts.
    """

    def __init__(self, device: Battery) -> None:
        super().__init__(device)
        self.energy_stored_upper: float = device.energy_capacity


class SocketDeviceSchedulePolicy(DeviceSchedulePolicy):
    """
    Class for fields determining the properties of the variables used when scheduling sockets based on power forecasts.
    """

    def __init__(self, device: SocketDevice, delta_t: int | float) -> None:
        super().__init__(device)
        energy_stored_buffer_upper: float = 0.9 * device.max_power_usage * delta_t
        energy_stored_buffer_lower: float = (
            device.max_power_usage * delta_t - energy_stored_buffer_upper
        )
        self.energy_stored_upper: float = (
            device.energy_capacity + energy_stored_buffer_upper
        )
        self.energy_considered_full: float = (
            device.energy_capacity - energy_stored_buffer_lower
        )
