from math import ceil
import pandas as pd
from control_homewizard_devices.device_classes import (
    SocketDevice,
    Battery,
    AggregateBattery,
)
from control_homewizard_devices.utils import ColNames
from typing import Any, cast


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
        self.socket_list.sort(key=lambda d: (d.priority, -d.max_power_usage))
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
        self.optional_socket_list = [
            device for device in self.socket_list if not device.daily_need
        ]

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
        self.df_variables = pd.DataFrame(
            index=data.df_power_interpolated.index, dtype=float
        )
        for device in data.devices_list:
            # in schedule: a device with -1 provides power (e.g. a battery discharging),
            # 0 a device does nothing, 1 a device consumes power
            self.df_variables[ColNames.state(device)] = 0.0
            self.df_variables[ColNames.energy_stored(device)] = device.energy_stored

        self.df_variables[ColNames.AVAILABLE_POWER] = data.df_power_interpolated[
            ColNames.POWER_W
        ].copy()


class DeviceSchedulingOptimization:
    """
    Class to handle the scheduling of devices based on power forecasts.
    """

    def __init__(
        self,
        delta_t: float = 0.25,
    ):
        self.delta_t = delta_t

    def solve_schedule_devices(
        self,
        df_power: pd.DataFrame,
        socket_and_battery_list: list[SocketDevice | Battery],
        overcharge: bool = False,
    ) -> tuple[ScheduleData, list[Variables]]:
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        results: list[Variables] = []
        self.data = ScheduleData(df_power, self.delta_t, socket_and_battery_list)
        self.variables = Variables(self.data)
        # Predicted power -->
        # Available power taking into account already scheduled devices.

        # Potential options for scheduling
        # 1. Schedule if enough available power.
        # 2. Enough available when batteries are added.
        # 3. Taking power from the grid.

        # Needed devices can use 1, 2 and 3.
        # Optional devices can only use 1

        for device in self.data.needed_socket_list:
            # Determine minimum number of timesteps needed to charge the device.
            charge_duration = self.get_charge_duration(device)

            # 1. Schedule if there is enough available power
            # Determine the timesteps where the device can be charged
            device_full = self.schedule_device_available_power(device, charge_duration)
            # If there are a sufficient number of timesteps --> go to the next device
            if device_full:
                continue
            # Otherwise try option 2.
            # 2. Schedule if the batteries can be used to schedule the devices
            # without using power from the grid.
            # --- Find the timesteps where the battery can be used
            # to provide enough power for the device ---
            if len(self.data.battery_list) > 0:
                device_full = self.schedule_device_with_battery(device, charge_duration)
                if device_full:
                    continue
            # Continue to option 3 if the device is not yet charged completely.
            # 3. Schedule device at maximum available power.
            self.schedule_max_power(device, charge_duration)
            results.append(self.variables)

        for device in self.data.optional_socket_list:
            # Determine minimum number of timesteps needed to charge the device.
            charge_duration = self.get_charge_duration(device)
            # Schedule if there is enough available power
            device_full = self.schedule_device_available_power(device, charge_duration)

        if overcharge:
            # Schedule devices as long as there is enough power available
            # Note: If a device consumes power while the program thinks it is full,
            # it is likely that the device has emptied slightly.
            # This can then be detected and allow the device to be charged until full.
            # Be aware that needed devices being detected empty
            # late into the day would cause them to demand power until full again.

            # To ensure each device is charged as much as possible
            # charge duration is set to the length of the dataframe
            max_duration = len(self.variables.df_variables.index)
            for device in self.data.socket_list:
                # TODO: Perhaps reduce energy stored in devices
                # To ensure that the energy stored in a device does not go to zero.
                self.schedule_device_available_power(device, max_duration)

        # Charge the aggregate battery at the end of the schedule.
        if len(self.data.battery_list) > 0:
            self.charge_batteries_remaining()

        results.append(self.variables)
        return self.data, results

    def get_charge_duration(self, device: SocketDevice) -> int:
        """
        Calculate the number of timesteps needed
        to charge the device to its full capacity.
        """
        return int(
            ceil(
                (device.policy.energy_considered_full - device.energy_stored)
                / (device.max_power_usage * self.delta_t)
            )
        )

    def schedule_device_available_power(
        self, device: SocketDevice, charge_duration: int
    ) -> bool:
        df_variables = self.variables.df_variables
        chargeable_mask = (
            df_variables[ColNames.AVAILABLE_POWER] >= device.max_power_usage
        )
        device_full = chargeable_mask.sum() >= charge_duration
        if device_full:
            # Schedule the device starting with the first available timesteps.
            indices = df_variables.index[chargeable_mask][:charge_duration]
        else:
            indices = df_variables.index[chargeable_mask]

        df_variables.loc[indices, ColNames.state(device)] = 1
        # Update available power.
        df_variables.loc[indices, ColNames.AVAILABLE_POWER] -= device.max_power_usage
        increased_energy = pd.Series(0.0, index=df_variables.index)
        increased_energy.loc[indices] = device.max_power_usage * self.delta_t
        cumsum_energy = increased_energy.cumsum()
        df_variables[ColNames.energy_stored(device)] += cumsum_energy
        return device_full

    def schedule_device_with_battery(
        self, device: SocketDevice, charge_duration: int
    ) -> bool:
        df_variables = self.variables.df_variables
        # Starting from timestep t=0 for each t:
        df_variables_battery_charged = df_variables.copy()
        max_charging_timesteps = 0.0
        min_diff = float("inf")
        best_temp_df_variables = df_variables.copy()
        for t_s in range(self.data.number_timesteps):
            # Charge the batteries at timestep t_s
            # (previous iterations ensure that the batteries are charged up to t_s)
            self.charge_batteries_at_timestep(t_s, df_variables_battery_charged)
            temp_df_variables = df_variables_battery_charged.copy()
            # Create a numpy array of available power for the scenario
            # where the batteries only charge until t_s; stopping afterwards.
            available_power = temp_df_variables[ColNames.AVAILABLE_POWER].to_numpy(
                copy=True
            )
            # available_power[:t_s] = available_power_battery_consumption[:t_s]
            for t_e in range(t_s + 1, self.data.number_timesteps):
                needed_power = device.max_power_usage - available_power[t_e]
                energy_stored = temp_df_variables.iat[
                    t_e - 1,
                    cast(
                        int,
                        temp_df_variables.columns.get_loc(
                            ColNames.energy_stored(self.data.aggregate_battery)
                        ),
                    ),
                ]
                # If batteries cannot provide enough power
                # or the device is already scheduled -->
                # Charge/discharge the batteries during this step
                if (
                    needed_power * self.delta_t > energy_stored
                    or needed_power > self.data.aggregate_battery.max_power_usage
                    or cast(
                        float,
                        temp_df_variables.iat[
                            t_e,
                            cast(
                                int,
                                temp_df_variables.columns.get_loc(
                                    ColNames.state(device)
                                ),
                            ),
                        ],
                    )
                    > 0
                ):
                    if available_power[t_e] < 0:
                        # If available power is negative, the battery discharges.
                        self.discharge_batteries_at_timestep(
                            -available_power[t_e], t_e, temp_df_variables
                        )
                    else:
                        self.charge_batteries_at_timestep(t_e, temp_df_variables)
                    continue

                # If battery has enough charge to provide the needed power
                # (device.max_power_usage - available power) --> schedule the device

                self.discharge_batteries_at_timestep(
                    needed_power, t_e, temp_df_variables
                )
                self.schedule_device(t_e, device, temp_df_variables)
            # If enough timesteps are available --> continue to the next device.
            charging_timesteps = temp_df_variables[ColNames.state(device)].sum()
            bool_schedule = temp_df_variables[ColNames.state(device)] > 0
            if bool_schedule.any():
                first_pos = bool_schedule.to_numpy().argmax()
                last_pos = (
                    len(bool_schedule) - 1 - bool_schedule.to_numpy()[::-1].argmax()
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

        # Update the schedule with the best scenario found.
        df_variables.update(best_temp_df_variables)
        # If the device could be scheduled completely --> go to next device.
        return max_charging_timesteps >= charge_duration

    def charge_batteries_at_timestep(self, timestep: int, df_schedule: pd.DataFrame):
        available_power: float = cast(
            float,
            df_schedule.iat[
                timestep,
                cast(int, df_schedule.columns.get_loc(ColNames.AVAILABLE_POWER)),
            ],
        )
        if available_power <= 0:
            return
        if timestep > 0:
            start_energy: float = cast(
                float,
                df_schedule.iat[
                    timestep - 1,
                    cast(
                        int,
                        df_schedule.columns.get_loc(
                            ColNames.energy_stored(self.data.aggregate_battery)
                        ),
                    ),
                ],
            )
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

    def discharge_batteries_at_timestep(
        self, needed_power: float, timestep: int, df_schedule: pd.DataFrame
    ):
        energy_stored = cast(
            float,
            df_schedule.at[
                df_schedule.index[timestep - 1],
                ColNames.energy_stored(self.data.aggregate_battery),
            ],
        )
        available_power = cast(
            float, df_schedule.at[df_schedule.index[timestep], ColNames.AVAILABLE_POWER]
        )

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
        energy_stored = cast(
            float,
            df_schedule.at[
                df_schedule.index[timestep - 1],
                ColNames.energy_stored(device),
            ],
        )
        available_power = cast(
            float, df_schedule.at[df_schedule.index[timestep], ColNames.AVAILABLE_POWER]
        )
        columns = [
            ColNames.state(device),
            ColNames.energy_stored(device),
            ColNames.AVAILABLE_POWER,
        ]
        new_values = [
            1.0,
            energy_stored + device.max_power_usage * self.delta_t,
            available_power - device.max_power_usage,
        ]
        df_schedule.loc[df_schedule.index[timestep], columns] = new_values

    def schedule_max_power(self, device: SocketDevice, charge_duration: int):
        df_variables = self.variables.df_variables
        # Since socket states should always be 0 or 1, we can convert it to int.
        already_charged = int(df_variables[ColNames.state(device)].sum())
        remaining_charge_duration = charge_duration - already_charged
        if remaining_charge_duration <= 0:
            return

        top_indices = (
            df_variables[ColNames.AVAILABLE_POWER]
            .nlargest(remaining_charge_duration)
            .index
        )
        df_variables.loc[top_indices, ColNames.state(device)] = 1
        # Update available power.
        df_variables.loc[top_indices, ColNames.AVAILABLE_POWER] -= (
            device.max_power_usage
        )
        increased_energy = pd.Series(0.0, index=df_variables.index)
        increased_energy.loc[top_indices] = device.max_power_usage * self.delta_t
        cumsum_energy = increased_energy.cumsum()
        df_variables[ColNames.energy_stored(device)] += cumsum_energy

    def charge_batteries_remaining(self):
        """
        Charge and uncharge the batteries with the remaining available power.
        (Charges in case available power is positive,
        uncharges in case available power is negative.)
        This is called after all devices have been scheduled.
        """
        df_variables = self.variables.df_variables
        mask = (
            (df_variables[ColNames.state(self.data.aggregate_battery)] == 0)
            & (df_variables[ColNames.AVAILABLE_POWER] != 0)
            & (
                df_variables[ColNames.energy_stored(self.data.aggregate_battery)]
                < self.data.aggregate_battery.max_energy_stored
            )
        )
        # Power consumed (capped at max_power_usage)
        possible_power_consumed = df_variables.loc[mask, ColNames.AVAILABLE_POWER].clip(
            lower=-self.data.aggregate_battery.max_power_usage,
            upper=self.data.aggregate_battery.max_power_usage,
        )

        # Update energy stored (capped at max_energy_stored)
        increased_energy = pd.Series(0.0, index=df_variables.index)
        increased_energy.loc[mask] = possible_power_consumed * self.delta_t
        cumsum_energy = increased_energy.cumsum()
        new_energy_stored_unlimited = (
            df_variables.loc[mask, ColNames.energy_stored(self.data.aggregate_battery)]
            + cumsum_energy
        )
        df_variables.loc[mask, ColNames.energy_stored(self.data.aggregate_battery)] = (
            new_energy_stored_unlimited.clip(
                lower=0, upper=self.data.aggregate_battery.max_energy_stored
            )
        )
        # Calculate the actual power consumed
        actual_power_consumed = (
            df_variables.loc[mask, ColNames.energy_stored(self.data.aggregate_battery)]
            - df_variables.loc[
                mask, ColNames.energy_stored(self.data.aggregate_battery)
            ]
            .shift(1)
            .fillna(0)
        ) / self.delta_t

        # Update available power
        df_variables.loc[mask, ColNames.AVAILABLE_POWER] = actual_power_consumed

        # Update state (capped at 1)
        new_state = (
            actual_power_consumed / self.data.aggregate_battery.max_power_usage
        ).clip(lower=-1, upper=1)
        df_variables.loc[mask, ColNames.state(self.data.aggregate_battery)] = new_state

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
    for result in results:
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
