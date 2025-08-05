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
from pyscipopt import Model, quicksum, Heur
from typing import Any
import numpy as np

PHASE_NUMBER_TO_DESCRIPTION_MAP = {
    0: "Power imbalance",
    1: "Switching count",
    2: "Early scheduling",
}


class ScheduleData:
    def __init__(
        self,
        df_power: pd.DataFrame,
        delta_t: float,
        socket_and_battery_list: list[SocketDevice | Battery],
    ) -> None:
        self.df_power_interpolated, self.P_p, self.number_timesteps = (
            self.preprocess_power_data(df_power, delta_t)
        )
        self.socket_and_battery_list = socket_and_battery_list
        self.socket_list = [
            device
            for device in socket_and_battery_list
            if isinstance(device, SocketDevice)
        ]
        self.needed_socket_list = [
            device for device in self.socket_list if device.daily_need
        ]
        self.MAX_PRIORITY = (
            max(device.priority for device in socket_and_battery_list) + 1
        )

    def preprocess_power_data(self, df_power: pd.DataFrame, delta_t: float):
        """
        Preprocess the power data by reindexing and interpolating.
        """
        new_datetime_index = pd.date_range(
            start=df_power.index.min(),
            end=df_power.index.max(),
            freq=f"{delta_t}h",
        )
        df_power_interpolated = df_power.reindex(new_datetime_index).interpolate(
            method="time"
        )
        P_p = (df_power_interpolated["power_kw"] * 1000).to_list()
        number_timesteps = len(P_p)
        return df_power_interpolated, P_p, number_timesteps


class SCIPVariables:
    """
    Class to hold and create my_variables for the scheduler.
    The model will have an instance of this class.
    The scheduler will in turn have a model instance.
    """

    def __init__(self, model: Model, data: ScheduleData) -> None:
        """
        Create my_variables for each device and time step.
        """
        # my_variables
        # in schedule: a device with -1 provides power (e.g. a battery discharging), 0 a device does nothing, 1 a device consumes power
        # The batteries can charge and discharge up to their max power, but can also do in between, therefore we should turn LpInteger into LpContinuous.
        self.schedule = {
            (device.device_name, t): model.addVar(
                name=f"O_{device.device_name}_{t}",
                lb=device.policy.schedule_lower,
                ub=device.policy.schedule_upper,
                vtype=device.policy.schedule_variable_cat.to_scip(),
            )
            for device in data.socket_and_battery_list
            for t in range(data.number_timesteps)
        }
        # Energy stored
        # The devices connected through the sockets often stop using power when they are full,
        # so we allow them to overcharge for up to 0.9 of the extra power consumed during a time step.
        # If we did not use the 0.9 and 0.1, the devices would be overcharged an for an entire extra step if the capacity and storage matched perfectly.
        self.E_s = {
            (device.device_name, t): model.addVar(
                name=f"E_s_{device.device_name}_{t}",
                lb=device.policy.energy_stored_lower,
                ub=device.policy.energy_stored_upper,
                vtype=SCIP_CONTINUOUS,
            )
            for device in data.socket_and_battery_list
            for t in range(data.number_timesteps)
        }
        self.grid_mode = {
            t: model.addVar(name=f"grid_mode_{t}", vtype=SCIP_BINARY)
            for t in range(data.number_timesteps)
        }
        self.grid_draw = {
            t: model.addVar(name=f"grid_draw_{t}", lb=0, ub=None, vtype=SCIP_CONTINUOUS)
            for t in range(data.number_timesteps)
        }
        self.grid_feed = {
            t: model.addVar(name=f"grid_feed_{t}", lb=0, ub=None, vtype=SCIP_CONTINUOUS)
            for t in range(data.number_timesteps)
        }
        # Variable to compensate failing to reach the energy stored for needed devices
        self.missing_energy = {
            device.device_name: model.addVar(
                name=f"missing_energy_{device.device_name}",
                lb=0,
                ub=None,
                vtype=SCIP_CONTINUOUS,
            )
            for device in data.needed_socket_list
        }
        # Objective variable to minimize
        self.z = model.addVar(name="z")
        # Create my_variables for the secondary objective of minimizing the number of times sockets are switched on/off
        self.switches = {
            (device.device_name, t): model.addVar(
                name=f"switch_{device.device_name}_{t}", vtype=SCIP_BINARY
            )
            for device in data.socket_list
            for t in range(data.number_timesteps - 1)
        }
        self.diff = {
            (device.device_name, t): model.addVar(
                name=f"diff_{device.device_name}_{t}",
                lb=device.policy.diff_lower,
                ub=device.policy.diff_upper,
                vtype=device.policy.diff_variable_cat.to_scip(),
            )
            for device in data.socket_list
            for t in range(data.number_timesteps - 1)
        }


@dataclass
class ScheduleResultSCIP:
    status: Any
    objective_value: float
    grid_draw: dict[int, float]
    grid_feed: dict[int, float]
    schedule: dict[tuple[str, int], float]
    energy_stored: dict[tuple[str, int], float]
    missing_energy: dict[str, float]

    @classmethod
    def from_vars(cls, vars_obj: SCIPVariables, model: Model, sol=None):
        if sol is None:
            sol = model.getBestSol()

        def extract(d):
            return {k: model.getSolVal(sol, v) for k, v in d.items()}

        return cls(
            model.getStatus(),
            model.getSolObjVal(sol),
            extract(vars_obj.grid_draw),
            extract(vars_obj.grid_feed),
            extract(vars_obj.schedule),
            extract(vars_obj.E_s),
            extract(vars_obj.missing_energy),
        )


class DeviceSchedulingOptimizationSCIP:
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
        time_limit: int = 60,
    ) -> tuple[ScheduleData, list[ScheduleResultSCIP]]:
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        model = Model()
        results: list[ScheduleResultSCIP] = []
        self.data = ScheduleData(df_power, self.delta_t, socket_and_battery_list)
        self.variables = SCIPVariables(model, self.data)

        missing_energy_penalty = quicksum(
            ENERGY_STORED_PENALTY * self.variables.missing_energy[device.device_name]
            for device in self.data.needed_socket_list
        )
        # Objective: minimize max absolute power imbalance
        combined_main_objective = self.variables.z + missing_energy_penalty
        model.setObjective(combined_main_objective, sense="minimize")

        # --- Constraints for first objective ---
        for t in range(self.data.number_timesteps):
            power_optional = quicksum(
                self.variables.schedule[device.device_name, t] * device.max_power_usage
                for device in self.data.socket_and_battery_list
                if not device.daily_need
            )
            power_needed = quicksum(
                self.variables.schedule[device.device_name, t] * device.max_power_usage
                for device in self.data.socket_and_battery_list
                if device.daily_need
            )
            # Power balance
            model.addCons(
                self.data.P_p[t] + self.variables.grid_draw[t]
                == power_optional + power_needed + self.variables.grid_feed[t],
                name="power_balance",
            )
            # grid draw and grid feed are mutually exclusive
            # A gridmode of 1 means that draw is possible and feed is not, and vice versa for a gridmode of 0.
            model.addCons(
                self.variables.grid_draw[t]
                <= MAX_GRID_DRAW * self.variables.grid_mode[t],
                name="grid_draw_mode",
            )
            model.addCons(
                self.variables.grid_feed[t]
                <= MAX_GRID_DRAW * (1 - self.variables.grid_mode[t]),
                name="grid_feed_mode",
            )
            # If grid mode is 1 (drawing from the grid), the optional devices cannot use any power.
            # If the grid mode is 0 (feeding to the grid), the optional devices can use power
            # (should not actually be limited by MAX_GRID_DRAW, but should be limited earlier through the power balance equation).
            model.addCons(
                power_optional <= MAX_GRID_DRAW * (1 - self.variables.grid_mode[t]),
                name="max_optional_power",
            )

            model.addCons(
                self.variables.grid_draw[t] <= self.variables.z,
                name="limit_grid_draw",
            )
            model.addCons(
                self.variables.grid_feed[t] <= self.data.P_p[t],
                name="limit_grid_feed",
            )

        # The preference is to feed more to the grid at certain times, such that devices can be on for a while and then off.
        model.addCons(
            quicksum(
                self.variables.grid_feed[t] for t in range(self.data.number_timesteps)
            )
            / self.data.number_timesteps
            <= self.variables.z
        )

        for device in self.data.socket_and_battery_list:
            for t in range(self.data.number_timesteps):
                if t == 0:
                    model.addCons(
                        self.variables.E_s[device.device_name, t]
                        == device.energy_stored
                        + self.variables.schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )
                else:
                    model.addCons(
                        self.variables.E_s[device.device_name, t]
                        == self.variables.E_s[device.device_name, t - 1]
                        + self.variables.schedule[device.device_name, t]
                        * device.max_power_usage
                        * self.delta_t
                    )

            if isinstance(device, SocketDevice) and device.daily_need:
                model.addCons(
                    self.variables.E_s[
                        device.device_name, self.data.number_timesteps - 1
                    ]
                    + self.variables.missing_energy[device.device_name]
                    >= device.policy.energy_considered_full
                )

        # --- constraints for second objective ---
        # Constraints needed to be added earlier to prevent diff/switches my_variables from causing an infeasible solution.
        for device in self.data.socket_list:
            for t in range(self.data.number_timesteps - 1):
                model.addCons(
                    self.variables.diff[device.device_name, t]
                    == self.variables.schedule[device.device_name, t + 1]
                    - self.variables.schedule[device.device_name, t]
                )
                model.addCons(
                    self.variables.diff[device.device_name, t]
                    <= device.policy.diff_upper
                    * self.variables.switches[device.device_name, t]
                )
                model.addCons(
                    self.variables.diff[device.device_name, t]
                    >= device.policy.diff_lower
                    * self.variables.switches[device.device_name, t]
                )
        # Solve the first objective
        model.setParam("limits/time", time_limit)
        model.optimize()
        opt_power_balance_value = model.getObjVal()

        # Add result to all results
        results.append(ScheduleResultSCIP.from_vars(self.variables, model))

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
            self.variables.switches[device.device_name, t]
            for t in range(self.data.number_timesteps - 1)
            for device in self.data.socket_list
        )
        model.setObjective(number_switches, "minimize")
        model.setParam("limits/time", time_limit)
        model.optimize()
        opt_number_switches = model.getObjVal()
        # Add result to all results
        results.append(ScheduleResultSCIP.from_vars(self.variables, model))

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
                (self.data.MAX_PRIORITY - device.priority)
                * t
                * self.variables.schedule[device.device_name, t]
                for t in range(self.data.number_timesteps)
                for device in self.data.socket_and_battery_list
            ),
            "minimize",
        )
        model.setParam("limits/time", time_limit)
        model.optimize()
        # prob.solve(PULP_CBC_CMD(msg=False))

        # Update the DataFrame with the schedules
        self.data.df_power_interpolated = self.update_dataframe(
            self.variables.schedule, self.data, model
        )

        # Print results
        results.append(ScheduleResultSCIP.from_vars(self.variables, model))

        return self.data, results

    def update_dataframe(
        self,
        schedule: dict[tuple[str, int], Any],
        data: ScheduleData,
        model: Model,
    ):
        """
        Update the DataFrame with the schedules of each device.
        """
        for device in data.socket_and_battery_list:
            self.data.df_power_interpolated[f"schedule {device.device_name}"] = 0.0
            for t, datetime_index in enumerate(data.df_power_interpolated.index):
                data.df_power_interpolated.at[
                    datetime_index, f"schedule {device.device_name}"
                ] = model.getVal(schedule[device.device_name, t])
        return data.df_power_interpolated

    def print_results(
        self, schedule_data: ScheduleData, results: list[ScheduleResultSCIP]
    ):
        """
        Print the results of the optimization.
        """
        # Prepare columns
        columns = (
            ["Time Step", "Available power (W)", "Grid Draw (W)", "Grid Feed (W)"]
            + [
                f"Schedule {d.device_name}"
                for d in schedule_data.socket_and_battery_list
            ]
            + [f"E_s {d.device_name}" for d in schedule_data.socket_and_battery_list]
        )
        for phase_number, result in enumerate(results):
            data = []
            for t in range(schedule_data.number_timesteps):
                row = [
                    t,
                    schedule_data.P_p[t],
                    result.grid_draw[t],
                    result.grid_feed[t],
                ]
                # Schedules
                row += [
                    result.schedule.get((device.device_name, t), 0.0)
                    for device in schedule_data.socket_and_battery_list
                ]
                # Energy stored
                row += [
                    result.energy_stored.get((device.device_name, t), 0.0)
                    for device in schedule_data.socket_and_battery_list
                ]
                data.append(row)

            df = pd.DataFrame(data, columns=columns)
            print("Status:", result.status)
            print(
                f"objective {PHASE_NUMBER_TO_DESCRIPTION_MAP[phase_number]}:",
                result.objective_value,
            )
            print(df.to_string(index=False, float_format="%.3f"))

            for device in schedule_data.needed_socket_list:
                print(
                    f"{device.device_name} has missing energy: {result.missing_energy[device.device_name]:.3f}"
                )


class GreedySchedulingHeuristic(Heur):
    def __init__(self, variables: SCIPVariables, data: ScheduleData):
        self.called = False
        self.variables = variables
        self.data = data

    def heurexec(self, heurtiming, nodeinfeasible):
        if self.called:
            return {"result": self.SCIP_DIDNOTRUN}
        self.called = True

        model = self.model
        sol = model.createSol(self)

        # Example: Get and set variable values in PySCIPOpt
        # To get a value: model.getVal(var) or model.getSolVal(sol, var)
        # To set a value: model.setSolVal(sol, var, value)

        grid_draw_np = np.array(
            [
                model.getSolVal(sol, self.variables.grid_draw[t])
                for t in range(self.data.number_timesteps)
            ]
        )
        grid_feed_np = np.array(
            [
                model.getSolVal(sol, self.variables.grid_draw[t])
                for t in range(self.data.number_timesteps)
            ]
        )

        # Greedy heuristic example: set all socket schedules to 1 if not full
        for device in self.data.socket_and_battery_list:
            for t in range(self.data.number_timesteps):
                var = self.variables.schedule[device.device_name, t]
                if isinstance(device, SocketDevice):
                    # Check if device is not full
                    energy_var = self.variables.E_s[device.device_name, t]
                    energy_val = model.getSolVal(sol, energy_var)
                    if energy_val < device.policy.energy_considered_full:
                        model.setSolVal(sol, var, 1.0)
                    else:
                        model.setSolVal(sol, var, 0.0)
                else:
                    # For batteries, set to max charge/discharge if possible
                    model.setSolVal(sol, var, model.getUbGlobal(var))

        # You can similarly set other variables as needed

        # Check and add solution
        if model.checkSol(sol):
            model.addSol(sol)
            return {"result": self.SCIP_FOUNDSOL}
        else:
            return {"result": self.SCIP_DIDNOTFIND}
