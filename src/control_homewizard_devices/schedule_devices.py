from collections import deque
from math import ceil
import pandas as pd
from dataclasses import dataclass
from control_homewizard_devices.device_classes import (
    SocketDevice,
    Battery,
    CompleteDevice,
)
from control_homewizard_devices.constants import (
    MAX_GRID_DRAW,
    ENERGY_STORED_PENALTY,
    EPSILON,
    SCIP_BINARY,
    SCIP_CONTINUOUS,
)
from typing import Any
import numpy as np
import heapq


class AggregateBattery:
    """
    Class to aggregate multiple batteries into one.
    This is used to simplify the scheduling process.
    """

    def __init__(self, battery_list: list[Battery]) -> None:
        self.device_name = "aggregate_battery"
        self.battery_list = battery_list
        self.max_power_usage = sum(battery.max_power_usage for battery in battery_list)
        self.energy_stored = sum(battery.energy_stored for battery in battery_list)
        self.max_energy_stored = sum(
            battery.policy.energy_stored_upper for battery in battery_list
        )


class ColNames:
    POWER_W = "power [w]"
    AVAILABLE_POWER = "available power"

    @staticmethod
    def state(device: CompleteDevice | AggregateBattery):
        return f"schedule {device.device_name}"

    @staticmethod
    def energy_stored(device: CompleteDevice | AggregateBattery):
        return f"energy stored {device.device_name}"


class ScheduleData:
    def __init__(
        self,
        df_power: pd.DataFrame,
        delta_t: float,
        socket_and_battery_list: list[SocketDevice | Battery],
    ) -> None:
        self.delta_t = delta_t
        self.df_power_interpolated, self.P_p, self.number_timesteps = (
            self.preprocess_power_data(df_power)
        )
        self.socket_and_battery_list = socket_and_battery_list
        self.socket_list = [
            device
            for device in socket_and_battery_list
            if isinstance(device, SocketDevice)
        ]
        self.battery_list = [
            device for device in socket_and_battery_list if isinstance(device, Battery)
        ]
        self.aggregate_battery = AggregateBattery(self.battery_list)
        if len(self.battery_list) > 0:
            self.devices_list = self.socket_list + [self.aggregate_battery]
        else:
            self.devices_list = self.socket_list
        self.needed_socket_list = [
            device for device in self.socket_list if device.daily_need
        ]
        self.MAX_PRIORITY = (
            max(device.priority for device in socket_and_battery_list) + 1
        )

    def preprocess_power_data(self, df_power: pd.DataFrame):
        """
        Preprocess the power data by reindexing and interpolating.
        """
        new_datetime_index = pd.date_range(
            start=df_power.index.min(),
            end=df_power.index.max(),
            freq=f"{self.delta_t}h",
        )
        df_power_interpolated = df_power.reindex(new_datetime_index).interpolate(
            method="time"
        )
        df_power_interpolated[ColNames.POWER_W] = (
            df_power_interpolated["power_kw"] * 1000
        )
        P_p: list[float] = (df_power_interpolated[ColNames.POWER_W]).to_list()
        number_timesteps = len(P_p)
        return df_power_interpolated, P_p, number_timesteps


class Variables:
    """
    Class to hold and create my_variables for the scheduler.
    The model will have an instance of this class.
    The scheduler will in turn have a model instance.
    """

    def __init__(self, data: ScheduleData) -> None:
        """
        Create my_variables for each device and time step.
        """
        self.df_variables = pd.DataFrame(index=data.df_power_interpolated.index)
        for device in data.devices_list:
            self.df_variables[ColNames.state(device)] = 0
            self.df_variables[ColNames.energy_stored(device)] = device.energy_stored

        self.df_variables[ColNames.AVAILABLE_POWER] = data.df_power_interpolated[
            ColNames.POWER_W
        ].copy()

        # my_variables
        # in schedule: a device with -1 provides power (e.g. a battery discharging), 0 a device does nothing, 1 a device consumes power
        # The batteries can charge and discharge up to their max power, but can also do in between, therefore we should turn LpInteger into LpContinuous.
        self.schedule = {
            (device.device_name, t): 0
            for device in data.socket_and_battery_list
            for t in range(data.number_timesteps)
        }
        # Energy stored
        # The devices connected through the sockets often stop using power when they are full,
        # so we allow them to overcharge for up to 0.9 of the extra power consumed during a time step.
        # If we did not use the 0.9 and 0.1, the devices would be overcharged an for an entire extra step if the capacity and storage matched perfectly.
        self.E_s = {
            (device.device_name, t): device.energy_stored
            for device in data.socket_and_battery_list
            for t in range(data.number_timesteps)
        }
        self.available_power = {t: data.P_p[t] for t in range(data.number_timesteps)}


class DeviceSchedulingOptimization:
    """
    Class to handle the scheduling of devices based on power forecasts.
    """

    def __init__(
        self,
        delta_t: float = 0.25,
    ):
        self.delta_t = delta_t

    def charge_batteries(self, timestep: int, df_schedule: pd.DataFrame):
        available_power = df_schedule.iat[
            timestep, df_schedule.columns.get_loc(ColNames.AVAILABLE_POWER)
        ]
        if available_power < 0:
            return
        if timestep > 0:
            start_energy: float = df_schedule.iat[
                timestep - 1,
                df_schedule.columns.get_loc(
                    ColNames.energy_stored(self.data.aggregate_battery)
                ),
            ]
        else:
            start_energy = self.data.aggregate_battery.energy_stored
        max_power = (
            self.data.aggregate_battery.max_energy_stored - start_energy
        ) / self.delta_t
        added_power = min(
            [self.data.aggregate_battery.max_power_usage, available_power, max_power]
        )
        columns = [
            ColNames.state(self.data.aggregate_battery),
            ColNames.energy_stored(self.data.aggregate_battery),
            ColNames.AVAILABLE_POWER,
        ]
        new_values = [
            added_power / self.data.aggregate_battery.max_power_usage,
            start_energy + added_power * self.delta_t,
            available_power - added_power,
        ]
        df_schedule.loc[df_schedule.index[timestep], columns] = new_values

    def discharge_batteries(
        self, needed_power: float, timestep: int, df_schedule: pd.DataFrame
    ):
        energy_stored = df_schedule.at[
            df_schedule.index[timestep],
            ColNames.energy_stored(self.data.aggregate_battery),
        ]
        available_power = df_schedule.at[
            df_schedule.index[timestep], ColNames.AVAILABLE_POWER
        ]

        possible_power = energy_stored / self.delta_t
        discharge_power = min(
            self.data.aggregate_battery.max_power_usage,
            possible_power,
            needed_power,
        )
        energy_stored -= discharge_power * self.delta_t
        columns = [
            ColNames.state(self.data.aggregate_battery),
            ColNames.energy_stored(self.data.aggregate_battery),
            ColNames.AVAILABLE_POWER,
        ]
        new_values = [
            -discharge_power / self.data.aggregate_battery.max_power_usage,
            energy_stored,
            available_power + discharge_power,
        ]
        df_schedule.loc[df_schedule.index[timestep], columns] = new_values

    def schedule_device(
        self, timestep: int, device: SocketDevice, df_schedule: pd.DataFrame
    ):
        energy_stored = df_schedule.at[
            df_schedule.index[timestep],
            ColNames.energy_stored(device),
        ]
        available_power = df_schedule.at[
            df_schedule.index[timestep], ColNames.AVAILABLE_POWER
        ]
        columns = [
            ColNames.state(self.data.aggregate_battery),
            ColNames.energy_stored(self.data.aggregate_battery),
            ColNames.AVAILABLE_POWER,
        ]
        new_values = [
            1.0,
            energy_stored + device.max_power_usage * self.delta_t,
            available_power - device.max_power_usage,
        ]
        df_schedule.loc[df_schedule.index[timestep], columns] = new_values

    def solve_schedule_devices(
        self,
        df_power: pd.DataFrame,
        socket_and_battery_list: list[SocketDevice | Battery],
        time_limit: int = 60,
    ) -> tuple[ScheduleData, list[Variables]]:
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        results: list[Variables] = []
        self.data = ScheduleData(df_power, self.delta_t, socket_and_battery_list)
        self.variables = Variables(self.data)
        df_variables = self.variables.df_variables

        # Predicted power --> Available power taking into account already scheduled devices.

        # Potential options for scheduling
        # 1. Schedule if enough available power.
        # 2. Enough available when batteries are added.
        # 3. Taking power from the grid.

        # Needed devices can use 1, 2 and 3.
        # Optional devices can only use 1

        for device in self.data.needed_socket_list:
            # Determine minimum number of timesteps needed to charge the device.
            charge_duration = int(
                ceil(device.policy.energy_considered_full - device.energy_stored)
                / (device.max_power_usage * self.delta_t)
            )

            # 1. Schedule if there is enough available power
            # Determine the timesteps where the device can be charged
            chargeable_mask = (
                df_variables[ColNames.AVAILABLE_POWER] > device.max_power_usage
            )
            device_full = chargeable_mask.sum() > charge_duration
            if device_full:
                # Schedule the device starting with the first available timesteps.
                indices = df_variables.index[chargeable_mask][:charge_duration]
            else:
                indices = df_variables.index[chargeable_mask]
            df_variables.loc[indices, ColNames.state(device)] = 1
            # Update available power.
            df_variables.loc[indices, ColNames.AVAILABLE_POWER] -= (
                device.max_power_usage
            )
            increased_energy = pd.Series(0.0, index=df_variables.index)
            increased_energy.loc[indices] = device.max_power_usage * self.delta_t
            cumsum_energy = increased_energy.cumsum()
            df_variables.loc[indices, ColNames.energy_stored(device)] += cumsum_energy

            # If there are a sufficient number of timesteps --> go to the next device
            if device_full:
                continue
            # Otherwise try option 2.
            # 2. Schedule if the batteries can be used to schedule the devices without using power from the grid.
            # --- Find the timesteps where the battery can be used to provide enough power for the device ---
            if len(self.data.battery_list) > 0:
                # Starting from timestep t=0 for each t:
                temp_df_variables = df_variables.copy()
                max_charging_timesteps = 0.0
                min_diff = float("inf")
                best_temp_df_variables = df_variables.copy()
                for t_s in range(self.data.number_timesteps):
                    # Charge the batteries at timestep t_s (previous iterations ensure that the batteries are charged at t_s)
                    self.charge_batteries(t_s, temp_df_variables)
                    # Create a numpy array of available power for the scenario where the batteries only charge until t_s; stopping afterwards.
                    available_power = temp_df_variables[
                        ColNames.AVAILABLE_POWER
                    ].to_numpy(copy=True)
                    # available_power[:t_s] = available_power_battery_consumption[:t_s]
                    for t_e in range(t_s, self.data.number_timesteps):
                        needed_power = device.max_power_usage - available_power[t_e]
                        energy_stored = temp_df_variables.iat[
                            t_e,
                            temp_df_variables.columns.get_loc(
                                ColNames.energy_stored(self.data.aggregate_battery)
                            ),
                        ]

                        # If batteries cannot provide enough power or the device is already scheduled --> Charge the batteries during this step
                        if (
                            needed_power * self.delta_t > energy_stored
                            or needed_power
                            > self.data.aggregate_battery.max_power_usage
                            or temp_df_variables.iat[
                                t_e,
                                temp_df_variables.columns.get_loc(
                                    ColNames.state(device)
                                ),
                            ]
                            > 0
                        ):
                            self.charge_batteries(t_e, temp_df_variables)
                            continue

                        # If battery has enough charge to provide the needed power (device.max_power_usage - available power) --> schedule the device
                        self.discharge_batteries(needed_power, t_e, temp_df_variables)
                        self.schedule_device(t_e, device, temp_df_variables)
                    # If enough timesteps are available --> continue to the next device.
                    charging_timesteps = temp_df_variables[ColNames.state(device)].sum()
                    bool_schedule = temp_df_variables[ColNames.state(device)] > 0
                    if bool_schedule.any():
                        first_pos = bool_schedule.to_numpy().argmax()
                        last_pos = (
                            len(bool_schedule)
                            - 1
                            - bool_schedule.to_numpy()[::-1].argmax()
                        )
                        diff = last_pos - first_pos
                    else:
                        diff = float("inf")
                    # Save the scenario with maximum timesteps available.
                    if charging_timesteps > max_charging_timesteps or (
                        charging_timesteps == max_charging_timesteps and diff < min_diff
                    ):
                        max_charging_timesteps = charging_timesteps
                        min_diff = diff
                        best_temp_df_variables = temp_df_variables.copy()
                    if max_charging_timesteps >= charge_duration:
                        # Best scenario found --> break
                        break

                # Update the schedule with the best scenario found.
                df_variables.update(best_temp_df_variables)
                if max_charging_timesteps >= charge_duration:
                    # If the device could be scheduled completely --> go to next device.
                    break
            # Continue to option 3 if the device is not yet charging completely.
            # 3. Schedule device at maximum available power.
            self.schedule_max_power(device)

        # TODO: Schedule the optional devices.
        sockets: list[SocketDevice] = self.data.needed_socket_list.copy()

        # Get needed devices last, then get the devices with largest power usage last, then get highest priority (smallest value) devices last
        sockets.sort(key=lambda d: (not d.daily_need, d.max_power_usage, -d.priority))
        results.append(self.variables)
        return self.data, results

    def schedule_max_power(self, device: SocketDevice):
        total_energy_stored = device.energy_stored
        # heapq orders by smallest value, so to get the largest value we use the negative
        initial_grid_feed = [
            (-self.variables.available_power[t], t)
            for t in range(self.data.number_timesteps)
        ]
        heapq.heapify(initial_grid_feed)
        while initial_grid_feed:
            (available_power, index) = heapq.heappop(initial_grid_feed)

            if self.variables.schedule[device.device_name, index] >= 0:
                # If device are already scheduled --> do not add this grid_feed back to the heapq.
                continue

            remaining_power = -available_power - device.max_power_usage
            heapq.heappush(initial_grid_feed, (-remaining_power, index))
            # Schedule the device at the index
            self.variables.schedule[device.device_name, index] = 1
            # Update total energy
            total_energy_stored += device.max_power_usage * self.data.delta_t
            self.variables.E_s[device.device_name, index] = (
                device.max_power_usage * self.data.delta_t
            )
            if total_energy_stored >= device.policy.energy_considered_full:
                # If device is now full --> break
                break

    def update_dataframe(
        self,
        schedule: dict[tuple[str, int], Any],
        data: ScheduleData,
    ):
        """
        Update the DataFrame with the schedules of each device.
        """
        for device in data.socket_and_battery_list:
            self.data.df_power_interpolated[f"schedule {device.device_name}"] = 0.0
            for t, datetime_index in enumerate(data.df_power_interpolated.index):
                data.df_power_interpolated.at[
                    datetime_index, f"schedule {device.device_name}"
                ] = schedule[device.device_name, t]
        return data.df_power_interpolated


def print_schedule_results(schedule_data: ScheduleData, results: list[Variables]):
    """
    Print the results of the optimization.
    """
    for phase_number, result in enumerate(results):
        print(result.df_variables.to_string(index=False, float_format="%.3f"))


def get_schedulable_socket(
    socket_list: list[SocketDevice], schedule: dict[tuple[str, int], int], index: int
):
    for i in reversed(range(len(socket_list))):
        device = socket_list[i]
        if schedule[device.device_name, index] <= 1:
            # Get the device that is not yet scheduled at this max.
            return device, i
    return None, None
