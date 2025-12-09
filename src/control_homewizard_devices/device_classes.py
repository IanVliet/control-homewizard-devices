from homewizard_energy import HomeWizardEnergy
from control_homewizard_devices.hwe_v2_wrapper.init_wrapper import HomeWizardEnergyV2
import logging
from dataclasses import dataclass
from time import time
from .constants import AGGREGATE_BATTERY

from control_homewizard_devices.constants import (
    DELTA_T,
    IS_FULL_POWER_RATIO,
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
        This method should be overridden in subclasses
        to implement specific measurement logic.

        For example, certain devices would also update the energy stored
        based on the instantaneous power usage or state of charge.
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
        For a general device, we assume the measured power directly indicates
        the available power, therefore we return power
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
        For the P1 a measured positive power already indicates the available power,
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
        min_battery_charge: float = 0.0,
        **kwargs,
    ):
        super().__init__(ip_address, device_type, device_name, **kwargs)
        self._max_power_usage = max_power_usage
        self.energy_capacity = energy_capacity
        self.policy = SocketDeviceSchedulePolicy(self, delta_t)
        self.priority = priority
        self.daily_need = daily_need
        self.min_battery_charge = min_battery_charge
        # the (instantaneous) attributes that change due to each measurement
        self.inst_state = None
        # whether the device should power on or off
        self.updated_state: bool | None = None
        self.energy_stored = 0.0
        self.time_last_update = time()
        # Use a in_control boolean flag that determines if:
        # 1. the schedule should be used to change the state
        # 2. or the device should be kept on until the device is full
        # This should enable users to turn a device on manually
        # until the device is full for special circumstances.
        self.in_control = True

    @property
    def max_power_usage(self):
        return self._max_power_usage

    async def perform_measurement(self, logger: logging.Logger):
        """
        Perform a measurement for the socket device.
        Gets the instantaneous power usage, current, and state of the socket.
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

            # Check whether the measured state is on while the updated state is off
            if self.updated_state is not None:
                if self.in_control and not self.updated_state and self.inst_state:
                    logger.info(
                        f"{self.device_name} has been turned on outside control."
                        " Setting in_control to False until device is full."
                    )
                    self.in_control = False

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
                # If the device was not in control, we can set it back to in control
                if not self.in_control:
                    self.in_control = True
                    logger.info(
                        f"{self.device_name} is fully charged. "
                        "Setting in_control to True."
                    )
            else:
                # Note: Updates energy stored with a measurement of the time passed
                # since the last update
                # (precision of energy stored of socket devices is limited)
                time_passed = time() - self.time_last_update
                logger.info(
                    f"{self.device_name} time passed since "
                    f"last state update: {time_passed:.3f} s"
                )
                self.energy_stored += self.inst_power_usage * time_passed / 3600
                # Perhaps limit energy stored to max capacity
                if self.energy_stored > self.energy_capacity:
                    self.energy_stored = self.energy_capacity
            logger.info(f"{self.device_name} energy stored: {self.energy_stored} Wh")

    def get_instantaneous_power(self):
        """
        For a socket a positive power indicates the power used by the socket,
        which can be made free by turning the socket off -->
        therefore this power should count to total available power.
        We define available power as the negative of the power measurement.
        Therefore, the function should return -power
        """
        if self.inst_power_usage is None:
            return None
        else:
            return -self.inst_power_usage

    def should_power_on(self, available_power: int | float):
        return self._max_power_usage <= available_power

    async def update_power_state(self, logger: logging.Logger):
        if self.hwe_device is not None and self.updated_state is not None:
            if self.in_control:
                await self.hwe_device.state_set(power_on=self.updated_state)
                self.time_last_update = time()
                logger.info(
                    f"{self.device_name} power state set to: {self.updated_state}"
                )
            else:
                logger.info(
                    f"{self.device_name} is not in control. "
                    "Power state will not be updated."
                )
        elif self.updated_state is None:
            logger.warning(f"{self.device_name}'s updated_state is None.")
        else:
            logger.warning(f"{self.device_name}'s hwe_device is None.")


@dataclass
class UserInfo:
    name: str
    token: str


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

    def get_instantaneous_power(self):
        """
        For a battery a positive power indicates the power used by the battery,
        which can be made free -->
        therefore this power should count to total available power.
        We define available power as the negative of the power measurement.
        Therefore, the function should return -power
        """
        if self.inst_power_usage is None:
            return None
        else:
            return -self.inst_power_usage


class DeviceSchedulePolicy:
    """
    Data class for fields determining the properties of the variables used
    when scheduling devices based on power forecasts.
    """

    def __init__(self, device: CompleteDevice) -> None:
        self.device = device
        self.energy_stored_upper: float


class BatterySchedulePolicy(DeviceSchedulePolicy):
    """
    Class for fields determining the properties of the variables used
    when scheduling batteries based on power forecasts.
    """

    def __init__(self, device: Battery) -> None:
        super().__init__(device)
        self.energy_stored_upper: float = device.energy_capacity


class SocketDeviceSchedulePolicy(DeviceSchedulePolicy):
    """
    Class for fields determining the properties of the variables used
    when scheduling sockets based on power forecasts.
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


class AggregateBattery:
    """
    Class to aggregate multiple batteries into one.
    This is used to simplify the scheduling process.
    """

    def __init__(self, battery_list: list[Battery]) -> None:
        self.device_name = AGGREGATE_BATTERY
        self.battery_list = battery_list
        self.max_power_usage = float(
            sum(battery.max_power_usage for battery in battery_list)
        )
        self.energy_stored = float(
            sum(battery.energy_stored for battery in battery_list)
        )
        self.max_energy_stored = sum(
            battery.policy.energy_stored_upper for battery in battery_list
        )
