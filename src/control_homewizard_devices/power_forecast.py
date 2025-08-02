from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime
import json
from plotly import graph_objects as go
import pandas as pd
from control_homewizard_devices.device_classes import SocketDevice, Battery
from dataclasses import dataclass
from typing import Any
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpInteger,
    LpContinuous,
    LpStatus,
    value,
    PULP_CBC_CMD,
)
from control_homewizard_devices.constants import (
    MAX_GRID_DRAW,
    ENERGY_STORED_PENALTY,
    EPSILON,
)


@dataclass
class ScheduleResult:
    """
    Class to hold the result of the scheduling optimization.
    Contains the DataFrame with the schedules and the optimization problem.
    """

    df_power_interpolated: pd.DataFrame
    prob: LpProblem
    combined_main_objective: Any
    number_timesteps: int
    grid_draw: dict[int, LpVariable]
    grid_feed: dict[int, LpVariable]
    P_p: list[float]
    schedule: dict[tuple[str, int], LpVariable]
    energy_stored: dict[tuple[str, int], LpVariable]
    missing_energy: dict[str, LpVariable]


class DeviceSchedulingOptimization:
    """
    Class to handle the scheduling of devices based on power forecasts.
    """

    def __init__(
        self,
        socket_and_battery_list: list[SocketDevice | Battery],
        delta_t: float = 0.25,
    ):
        self.socket_and_battery_list = socket_and_battery_list
        self.socket_list = [
            device
            for device in socket_and_battery_list
            if isinstance(device, SocketDevice)
        ]
        self.needed_socket_list = [
            device for device in self.socket_list if device.daily_need
        ]
        self.delta_t = delta_t
        self.MAX_PRIORITY = (
            max(device.priority for device in self.socket_and_battery_list) + 1
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
        P_p = (df_power_interpolated["power_kw"] * 1000).to_list()
        number_timesteps = len(P_p)
        return df_power_interpolated, P_p, number_timesteps

    def create_variables(self, number_timesteps: int):
        """
        Create variables for each device and time step.
        """
        # Variables
        # in schedule: a device with -1 provides power (e.g. a battery discharging), 0 a device does nothing, 1 a device consumes power
        # The batteries can charge and discharge up to their max power, but can also do in between, therefore we should turn LpInteger into LpContinuous.
        schedules = {
            (device.device_name, t): LpVariable(
                f"O_{device.device_name}_{t}",
                device.policy.schedule_lower,
                device.policy.schedule_upper,
                cat=device.policy.schedule_variable_cat.value,
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps)
        }
        # Energy stored
        # The devices connected through the sockets often stop using power when they are full,
        # so we allow them to overcharge for up to 0.9 of the extra power consumed during a time step.
        # If we did not use the 0.9 and 0.1, the devices would be overcharged an for an entire extra step if the capacity and storage matched perfectly.
        E_s = {
            (device.device_name, t): LpVariable(
                f"E_s_{device.device_name}_{t}",
                device.policy.energy_stored_lower,
                device.policy.energy_stored_upper,
                cat=LpContinuous,
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps)
        }
        grid_mode = {
            t: LpVariable(f"grid_mode_{t}", 0, 1, cat=LpBinary)
            for t in range(number_timesteps)
        }
        grid_draw = {
            t: LpVariable(f"grid_draw_{t}", 0, None, cat=LpContinuous)
            for t in range(number_timesteps)
        }
        grid_feed = {
            t: LpVariable(f"grid_feed_{t}", 0, None, cat=LpContinuous)
            for t in range(number_timesteps)
        }
        # Variable to compensate failing to reach the energy stored for needed devices
        missing_energy = {
            device.device_name: LpVariable(
                f"missing_energy_{device.device_name}", 0, None, cat=LpContinuous
            )
            for device in self.needed_socket_list
        }
        # Objective variable to minimize
        z = LpVariable("z")
        return schedules, E_s, grid_mode, grid_draw, grid_feed, missing_energy, z

    def solve_schedule_devices(
        self, df_power: pd.DataFrame, time_limit: int = 60
    ) -> pd.DataFrame:
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        prob = LpProblem("DeviceSchedulingOptimization", LpMinimize)

        df_power_interpolated, P_p, number_timesteps = self.preprocess_power_data(
            df_power
        )

        schedule, E_s, grid_mode, grid_draw, grid_feed, missing_energy, z = (
            self.create_variables(number_timesteps)
        )

        missing_energy_penalty = lpSum(
            ENERGY_STORED_PENALTY * missing_energy[device.device_name]
            for device in self.needed_socket_list
        )
        # Objective: minimize max absolute power imbalance
        combined_main_objective = z + missing_energy_penalty
        prob.setObjective(combined_main_objective)

        # Constraints
        for t in range(number_timesteps):
            power_optional = lpSum(
                schedule[device.device_name, t] * device.max_power_usage
                for device in self.socket_and_battery_list
                if not device.daily_need
            )
            power_needed = lpSum(
                schedule[device.device_name, t] * device.max_power_usage
                for device in self.socket_and_battery_list
                if device.daily_need
            )
            # Power balance
            prob += (
                P_p[t] + grid_draw[t] == power_optional + power_needed + grid_feed[t]
            )
            # grid draw and grid feed are mutually exclusive
            # A gridmode of 1 means that draw is possible and feed is not, and vice versa for a gridmode of 0.
            prob += grid_draw[t] <= MAX_GRID_DRAW * grid_mode[t]
            prob += grid_feed[t] <= MAX_GRID_DRAW * (1 - grid_mode[t])
            # If grid mode is 1 (drawing from the grid), the optional devices cannot use any power.
            # If the grid mode is 0 (feeding to the grid), the optional devices can use power
            # (should not actually be limited by MAX_GRID_DRAW, but should be limited earlier through the power balance equation).
            prob += power_optional <= MAX_GRID_DRAW * (1 - grid_mode[t])

            prob += grid_draw[t] <= z
            prob += grid_feed[t] <= P_p[t]

        # The preference is to feed more to the grid at certain times, such that devices can be on for a while and then off.
        prob += (
            lpSum(grid_feed[t] for t in range(number_timesteps)) / number_timesteps <= z
        )

        for device in self.socket_and_battery_list:
            for t in range(number_timesteps):
                if t == 0:
                    prob += (
                        E_s[device.device_name, t]
                        == device.energy_stored
                        + schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )
                else:
                    prob += (
                        E_s[device.device_name, t]
                        == E_s[device.device_name, t - 1]
                        + schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )

            if isinstance(device, SocketDevice) and device.daily_need:
                prob += (
                    E_s[device.device_name, number_timesteps - 1]
                    + missing_energy[device.device_name]
                    >= device.policy.energy_considered_full
                )
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        opt_power_balance_value = value(prob.objective)

        # Print results
        first_result = ScheduleResult(
            df_power_interpolated=df_power_interpolated,
            prob=prob,
            combined_main_objective=combined_main_objective,
            number_timesteps=number_timesteps,
            grid_draw=grid_draw,
            grid_feed=grid_feed,
            P_p=P_p,
            schedule=schedule,
            energy_stored=E_s,
            missing_energy=missing_energy,
        )
        # print("Initial optimization results:")
        # self.print_results(first_result)
        if opt_power_balance_value is None:
            raise ValueError("The optimization problem could not be solved.")
        # Solve the secondary objective of minimizing the number of times sockets are switched on/off
        switches = {
            (device.device_name, t): LpVariable(
                f"switch_{device.device_name}_{t}", 0, 1, cat=LpBinary
            )
            for device in self.socket_list
            for t in range(number_timesteps - 1)
        }
        diff = {
            (device.device_name, t): LpVariable(
                f"diff_{device.device_name}_{t}",
                device.policy.diff_lower,
                device.policy.diff_upper,
                cat=device.policy.diff_variable_cat.value,
            )
            for device in self.socket_list
            for t in range(number_timesteps - 1)
        }
        # Allow a small tolerance (epsilon) to avoid infeasibility in secondary objective
        prob += (
            combined_main_objective <= opt_power_balance_value + EPSILON,
            "FixPowerObjective",
        )

        for device in self.socket_list:
            for t in range(number_timesteps - 1):
                prob += (
                    diff[device.device_name, t]
                    == schedule[device.device_name, t + 1]
                    - schedule[device.device_name, t]
                )
                prob += (
                    diff[device.device_name, t]
                    <= device.policy.diff_upper * switches[device.device_name, t]
                )
                prob += (
                    diff[device.device_name, t]
                    >= device.policy.diff_lower * switches[device.device_name, t]
                )
        number_switches = lpSum(
            switches[device.device_name, t]
            for t in range(number_timesteps - 1)
            for device in self.socket_list
        )
        prob.setObjective(number_switches)
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        opt_number_switches = value(prob.objective)
        # Print results
        second_result = ScheduleResult(
            df_power_interpolated=df_power_interpolated,
            prob=prob,
            combined_main_objective=combined_main_objective,
            number_timesteps=number_timesteps,
            grid_draw=grid_draw,
            grid_feed=grid_feed,
            P_p=P_p,
            schedule=schedule,
            energy_stored=E_s,
            missing_energy=missing_energy,
        )
        # print("Secondary optimization results:")
        # self.print_results(second_result)

        # Solve the thirtiary objective of ensuring that devices are turned on as early as possible,
        # with device priority: lower priority number means higher priority (should be turned on earlier).
        prob += combined_main_objective == opt_power_balance_value, "FixPowerObjective2"
        prob += number_switches == opt_number_switches, "FixSwitchObjective"
        # To prioritize lower priority numbers (higher priority) to turn on earlier,
        # use a large constant minus the priority, so higher priority (lower number) gets a larger weight.

        prob.setObjective(
            lpSum(
                (self.MAX_PRIORITY - device.priority)
                * t
                * schedule[device.device_name, t]
                for t in range(number_timesteps)
                for device in self.socket_and_battery_list
            )
        )
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        # prob.solve(PULP_CBC_CMD(msg=False))

        # Update the DataFrame with the schedules
        df_power_interpolated = self.update_dataframe(schedule, df_power_interpolated)

        # Print results
        final_result = ScheduleResult(
            df_power_interpolated=df_power_interpolated,
            prob=prob,
            combined_main_objective=combined_main_objective,
            number_timesteps=number_timesteps,
            grid_draw=grid_draw,
            grid_feed=grid_feed,
            P_p=P_p,
            schedule=schedule,
            energy_stored=E_s,
            missing_energy=missing_energy,
        )
        self.print_results(final_result)

        return df_power_interpolated

    def update_dataframe(
        self,
        schedule: dict[tuple[str, int], LpVariable],
        df_power_interpolated: pd.DataFrame,
    ):
        """
        Update the DataFrame with the schedules of each device.
        """
        for device in self.socket_and_battery_list:
            df_power_interpolated[f"schedule {device.device_name}"] = 0.0
            for t, datetime_index in enumerate(df_power_interpolated.index):
                df_power_interpolated.at[
                    datetime_index, f"schedule {device.device_name}"
                ] = value(schedule[device.device_name, t])
        return df_power_interpolated

    def print_results(self, result: ScheduleResult):
        """
        Print the results of the optimization.
        """
        # Prepare columns
        columns = (
            ["Time Step", "Available power (W)", "Grid Draw (W)", "Grid Feed (W)"]
            + [f"Schedule {d.device_name}" for d in self.socket_and_battery_list]
            + [f"E_s {d.device_name}" for d in self.socket_and_battery_list]
        )

        data = []
        for t in range(result.number_timesteps):
            row = [
                t,
                result.P_p[t],
                value(result.grid_draw[t]),
                value(result.grid_feed[t]),
            ]
            # Schedules
            row += [
                value(result.schedule.get((device.device_name, t), 0.0))
                for device in self.socket_and_battery_list
            ]
            # Energy stored
            row += [
                value(result.energy_stored.get((device.device_name, t), 0.0))
                for device in self.socket_and_battery_list
            ]
            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        print("Status:", LpStatus[result.prob.status])
        print("Max imbalance:", value(result.combined_main_objective))
        print(df.to_string(index=False, float_format="%.3f"))

        for device in self.needed_socket_list:
            print(
                f"{device.device_name} has missing energy: {value(result.missing_energy[device.device_name]):.3f}"
            )
        print("Optimization finished")
        print("Max imbalance:", value(result.combined_main_objective))


if __name__ == "__main__":
    with open("config_devices.json", "r") as config_file:
        config_data = json.load(config_file)
    solar_panels_data = config_data["solar panels"]
    for solar_panel_data in solar_panels_data:
        site = PVSite(
            latitude=solar_panel_data["latitude"],
            longitude=solar_panel_data["longitude"],
            capacity_kwp=solar_panel_data["total peak power"],
            tilt=solar_panel_data["tilt"],
            orientation=solar_panel_data["orientation"],
        )

        predictions_df = run_forecast(site=site, ts=datetime.today(), nwp_source="icon")
        predictions_df.to_csv("../data/solar_prediction.csv")
        predictions_df.to_parquet("../data/solar_prediction.parquet")
        fig = go.Figure(go.Scatter(y=predictions_df["power_kw"]))
        fig.show()
        break
