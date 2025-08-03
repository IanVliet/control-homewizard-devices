import pandas as pd
from dataclasses import dataclass
from control_homewizard_devices.device_classes import SocketDevice, Battery
from control_homewizard_devices.constants import (
    MAX_GRID_DRAW,
    ENERGY_STORED_PENALTY,
    EPSILON,
    SCIP_BINARY,
    SCIP_CONTINUOUS,
)
from pyscipopt import Model, quicksum
from typing import Any


@dataclass
class ScheduleResultSCIP:
    df_power_interpolated: pd.DataFrame
    model: Model
    main_objective_value: Any
    number_timesteps: int
    grid_draw: dict[int, Any]
    grid_feed: dict[int, Any]
    P_p: list[float]
    schedule: dict[tuple[str, int], Any]
    energy_stored: dict[tuple[str, int], Any]
    missing_energy: dict[str, Any]


class DeviceSchedulingOptimizationSCIP:
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

    def create_variables(self, number_timesteps: int, model: Model):
        """
        Create variables for each device and time step.
        """
        # Variables
        # in schedule: a device with -1 provides power (e.g. a battery discharging), 0 a device does nothing, 1 a device consumes power
        # The batteries can charge and discharge up to their max power, but can also do in between, therefore we should turn LpInteger into LpContinuous.
        schedules = {
            (device.device_name, t): model.addVar(
                name=f"O_{device.device_name}_{t}",
                lb=device.policy.schedule_lower,
                ub=device.policy.schedule_upper,
                vtype=device.policy.schedule_variable_cat.to_scip(),
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps)
        }
        # Energy stored
        # The devices connected through the sockets often stop using power when they are full,
        # so we allow them to overcharge for up to 0.9 of the extra power consumed during a time step.
        # If we did not use the 0.9 and 0.1, the devices would be overcharged an for an entire extra step if the capacity and storage matched perfectly.
        E_s = {
            (device.device_name, t): model.addVar(
                name=f"E_s_{device.device_name}_{t}",
                lb=device.policy.energy_stored_lower,
                ub=device.policy.energy_stored_upper,
                vtype=SCIP_CONTINUOUS,
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps)
        }
        grid_mode = {
            t: model.addVar(name=f"grid_mode_{t}", vtype=SCIP_BINARY)
            for t in range(number_timesteps)
        }
        grid_draw = {
            t: model.addVar(name=f"grid_draw_{t}", lb=0, ub=None, vtype=SCIP_CONTINUOUS)
            for t in range(number_timesteps)
        }
        grid_feed = {
            t: model.addVar(name=f"grid_feed_{t}", lb=0, ub=None, vtype=SCIP_CONTINUOUS)
            for t in range(number_timesteps)
        }
        # Variable to compensate failing to reach the energy stored for needed devices
        missing_energy = {
            device.device_name: model.addVar(
                name=f"missing_energy_{device.device_name}",
                lb=0,
                ub=None,
                vtype=SCIP_CONTINUOUS,
            )
            for device in self.needed_socket_list
        }
        # Objective variable to minimize
        z = model.addVar(name="z")
        # Create variables for the secondary objective of minimizing the number of times sockets are switched on/off
        switches = {
            (device.device_name, t): model.addVar(
                name=f"switch_{device.device_name}_{t}", vtype=SCIP_BINARY
            )
            for device in self.socket_list
            for t in range(number_timesteps - 1)
        }
        diff = {
            (device.device_name, t): model.addVar(
                name=f"diff_{device.device_name}_{t}",
                lb=device.policy.diff_lower,
                ub=device.policy.diff_upper,
                vtype=device.policy.diff_variable_cat.to_scip(),
            )
            for device in self.socket_list
            for t in range(number_timesteps - 1)
        }
        return (
            schedules,
            E_s,
            grid_mode,
            grid_draw,
            grid_feed,
            missing_energy,
            z,
            switches,
            diff,
        )

    def solve_schedule_devices(
        self, df_power: pd.DataFrame, time_limit: int = 60
    ) -> list[ScheduleResultSCIP]:
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        model = Model()
        results: list[ScheduleResultSCIP] = []
        df_power_interpolated, P_p, number_timesteps = self.preprocess_power_data(
            df_power
        )
        (
            schedule,
            E_s,
            grid_mode,
            grid_draw,
            grid_feed,
            missing_energy,
            z,
            switches,
            diff,
        ) = self.create_variables(number_timesteps, model)

        missing_energy_penalty = quicksum(
            ENERGY_STORED_PENALTY * missing_energy[device.device_name]
            for device in self.needed_socket_list
        )
        # Objective: minimize max absolute power imbalance
        combined_main_objective = z + missing_energy_penalty
        model.setObjective(combined_main_objective, sense="minimize")

        # --- Constraints for first objective ---
        for t in range(number_timesteps):
            power_optional = quicksum(
                schedule[device.device_name, t] * device.max_power_usage
                for device in self.socket_and_battery_list
                if not device.daily_need
            )
            power_needed = quicksum(
                schedule[device.device_name, t] * device.max_power_usage
                for device in self.socket_and_battery_list
                if device.daily_need
            )
            # Power balance
            model.addCons(
                P_p[t] + grid_draw[t] == power_optional + power_needed + grid_feed[t],
                name="power_balance",
            )
            # grid draw and grid feed are mutually exclusive
            # A gridmode of 1 means that draw is possible and feed is not, and vice versa for a gridmode of 0.
            model.addCons(
                grid_draw[t] <= MAX_GRID_DRAW * grid_mode[t], name="grid_draw_mode"
            )
            model.addCons(
                grid_feed[t] <= MAX_GRID_DRAW * (1 - grid_mode[t]),
                name="grid_feed_mode",
            )
            # If grid mode is 1 (drawing from the grid), the optional devices cannot use any power.
            # If the grid mode is 0 (feeding to the grid), the optional devices can use power
            # (should not actually be limited by MAX_GRID_DRAW, but should be limited earlier through the power balance equation).
            model.addCons(
                power_optional <= MAX_GRID_DRAW * (1 - grid_mode[t]),
                name="max_optional_power",
            )

            model.addCons(grid_draw[t] <= z, name="limit_grid_draw")
            model.addCons(grid_feed[t] <= P_p[t], name="limit_grid_feed")

        # The preference is to feed more to the grid at certain times, such that devices can be on for a while and then off.
        model.addCons(
            quicksum(grid_feed[t] for t in range(number_timesteps)) / number_timesteps
            <= z
        )

        for device in self.socket_and_battery_list:
            for t in range(number_timesteps):
                if t == 0:
                    model.addCons(
                        E_s[device.device_name, t]
                        == device.energy_stored
                        + schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )
                else:
                    model.addCons(
                        E_s[device.device_name, t]
                        == E_s[device.device_name, t - 1]
                        + schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )

            if isinstance(device, SocketDevice) and device.daily_need:
                model.addCons(
                    E_s[device.device_name, number_timesteps - 1]
                    + missing_energy[device.device_name]
                    >= device.policy.energy_considered_full
                )

        # --- constraints for second objective ---
        # Constraints needed to be added earlier to prevent diff/switches variables from causing an infeasible solution.
        for device in self.socket_list:
            for t in range(number_timesteps - 1):
                model.addCons(
                    diff[device.device_name, t]
                    == schedule[device.device_name, t + 1]
                    - schedule[device.device_name, t]
                )
                model.addCons(
                    diff[device.device_name, t]
                    <= device.policy.diff_upper * switches[device.device_name, t]
                )
                model.addCons(
                    diff[device.device_name, t]
                    >= device.policy.diff_lower * switches[device.device_name, t]
                )
        # Solve the first objective
        model.setParam("limits/time", time_limit)
        model.optimize()
        opt_power_balance_value = model.getObjVal()

        # Add result to all results
        results.append(
            ScheduleResultSCIP(
                df_power_interpolated=df_power_interpolated,
                model=model,
                main_objective_value=opt_power_balance_value,
                number_timesteps=number_timesteps,
                grid_draw=grid_draw,
                grid_feed=grid_feed,
                P_p=P_p,
                schedule=schedule,
                energy_stored=E_s,
                missing_energy=missing_energy,
            )
        )

        if opt_power_balance_value is None:
            raise ValueError("The optimization problem could not be solved.")
        # self.print_results(results)

        model.freeTransform()
        model.resetParams()
        # Setup secondary objective
        # Allow a small tolerance (epsilon) to avoid infeasibility in secondary objective
        model.addCons(
            combined_main_objective <= opt_power_balance_value + EPSILON,
            "FixPowerObjective",
        )
        # Activate the second objective.
        number_switches = quicksum(
            switches[device.device_name, t]
            for t in range(number_timesteps - 1)
            for device in self.socket_list
        )
        model.setObjective(number_switches, "minimize")
        model.setParam("limits/time", time_limit)
        model.optimize()
        opt_number_switches = model.getObjVal()
        # Add result to all results
        results.append(
            ScheduleResultSCIP(
                df_power_interpolated=df_power_interpolated,
                model=model,
                main_objective_value=opt_power_balance_value,
                number_timesteps=number_timesteps,
                grid_draw=grid_draw,
                grid_feed=grid_feed,
                P_p=P_p,
                schedule=schedule,
                energy_stored=E_s,
                missing_energy=missing_energy,
            )
        )

        model.freeTransform()
        # Solve the thirtiary objective of ensuring that devices are turned on as early as possible,
        # with device priority: lower priority number means higher priority (should be turned on earlier).
        model.addCons(
            combined_main_objective == opt_power_balance_value, "FixPowerObjective2"
        )
        model.addCons(number_switches == opt_number_switches, "FixSwitchObjective")
        # To prioritize lower priority numbers (higher priority) to turn on earlier,
        # use a large constant minus the priority, so higher priority (lower number) gets a larger weight.

        model.setObjective(
            quicksum(
                (self.MAX_PRIORITY - device.priority)
                * t
                * schedule[device.device_name, t]
                for t in range(number_timesteps)
                for device in self.socket_and_battery_list
            ),
            "minimize",
        )
        model.setParam("limits/time", time_limit)
        model.optimize()
        # prob.solve(PULP_CBC_CMD(msg=False))

        # Update the DataFrame with the schedules
        df_power_interpolated = self.update_dataframe(
            schedule, df_power_interpolated, model
        )

        # Print results
        results.append(
            ScheduleResultSCIP(
                df_power_interpolated=df_power_interpolated,
                model=model,
                main_objective_value=opt_power_balance_value,
                number_timesteps=number_timesteps,
                grid_draw=grid_draw,
                grid_feed=grid_feed,
                P_p=P_p,
                schedule=schedule,
                energy_stored=E_s,
                missing_energy=missing_energy,
            )
        )

        return results

    def update_dataframe(
        self,
        schedule: dict[tuple[str, int], Any],
        df_power_interpolated: pd.DataFrame,
        model: Model,
    ):
        """
        Update the DataFrame with the schedules of each device.
        """
        for device in self.socket_and_battery_list:
            df_power_interpolated[f"schedule {device.device_name}"] = 0.0
            for t, datetime_index in enumerate(df_power_interpolated.index):
                df_power_interpolated.at[
                    datetime_index, f"schedule {device.device_name}"
                ] = model.getVal(schedule[device.device_name, t])
        return df_power_interpolated

    def print_results(self, results: list[ScheduleResultSCIP]):
        """
        Print the results of the optimization.
        """
        # Prepare columns
        columns = (
            ["Time Step", "Available power (W)", "Grid Draw (W)", "Grid Feed (W)"]
            + [f"Schedule {d.device_name}" for d in self.socket_and_battery_list]
            + [f"E_s {d.device_name}" for d in self.socket_and_battery_list]
        )
        for result in results:
            data = []
            for t in range(result.number_timesteps):
                row = [
                    t,
                    result.P_p[t],
                    result.model.getVal(result.grid_draw[t]),
                    result.model.getVal(result.grid_feed[t]),
                ]
                # Schedules
                row += [
                    result.model.getVal(
                        result.schedule.get((device.device_name, t), 0.0)
                    )
                    for device in self.socket_and_battery_list
                ]
                # Energy stored
                row += [
                    result.model.getVal(
                        result.energy_stored.get((device.device_name, t), 0.0)
                    )
                    for device in self.socket_and_battery_list
                ]
                data.append(row)

            df = pd.DataFrame(data, columns=columns)
            print("Status:", result.model.getStatus())
            print("Max imbalance:", result.main_objective_value)
            print(df.to_string(index=False, float_format="%.3f"))

            for device in self.needed_socket_list:
                print(
                    f"{device.device_name} has missing energy: {result.model.getVal(result.missing_energy[device.device_name]):.3f}"
                )
