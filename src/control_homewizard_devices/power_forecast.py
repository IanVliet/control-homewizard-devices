from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime
import json
from plotly import graph_objects as go
import pandas as pd
from control_homewizard_devices.device_classes import SocketDevice, Battery
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
)


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
        self.delta_t = delta_t

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
        # power (left) remaining variable (< 0 means power is taken from the grid, > 0 means power is sent to the grid)
        P_l = {t: LpVariable(f"P_l_{t}") for t in range(number_timesteps)}
        # Objective variable to minimize
        z = LpVariable("z")
        return schedules, E_s, P_l, z

    def preprocess_power_data(self, df_power: pd.DataFrame):
        """
        Preprocess the power data by reindexing and interpolating.
        """
        new_datetime_index = pd.date_range(
            start=df_power.index[0], end=df_power.index[-1], freq=f"{self.delta_t}h"
        )
        df_power_interpolated = df_power.reindex(new_datetime_index).interpolate(
            method="time"
        )
        P_p = (df_power_interpolated["power_kw"] * 1000).to_list()
        number_timesteps = len(P_p)
        return df_power_interpolated, P_p, number_timesteps

    def solve_schedule_devices(self, df_power: pd.DataFrame):
        """
        Solve the scheduling problem for devices that need to be charged.
        """
        # create the problem to be optimized
        prob = LpProblem("DeviceSchedulingOptimization", LpMinimize)

        df_power_interpolated, P_p, number_timesteps = self.preprocess_power_data(
            df_power
        )

        schedule, E_s, P_l, z = self.create_variables(number_timesteps)

        # Objective: minimize max absolute power imbalance
        prob += z

        # Constraints
        for t in range(number_timesteps):
            # Power left
            prob += P_l[t] == P_p[t] - lpSum(
                schedule[device.device_name, t] * device.max_power_usage
                for device in self.socket_and_battery_list
            )
            prob += -P_l[t] <= z
            prob += P_l[t] <= P_p[t]

        prob += lpSum(P_l[t] for t in range(number_timesteps)) / number_timesteps <= z

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
                    >= device.policy.energy_considered_full
                )

        # Solve the problem
        prob.solve()

        # Solve the secondary objective of minimizing the number of times sockets are switched on/off
        opt_value = value(prob.objective)
        switches = {
            (device.device_name, t): LpVariable(
                f"switch_{device.device_name}_{t}", 0, 1, cat=LpBinary
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps - 1)
        }
        diff = {
            (device.device_name, t): LpVariable(
                f"diff_{device.device_name}_{t}",
                device.policy.diff_lower,
                device.policy.diff_upper,
                cat=device.policy.diff_variable_cat.value,
            )
            for device in self.socket_and_battery_list
            for t in range(number_timesteps - 1)
        }
        prob += z == opt_value, "FixMainObjective"

        for device in self.socket_and_battery_list:
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
                    -diff[device.device_name, t]
                    <= -device.policy.diff_lower * switches[device.device_name, t]
                )

        prob.setObjective(
            lpSum(
                switches[device.device_name, t]
                for t in range(number_timesteps - 1)
                for device in self.socket_and_battery_list
            )
        )
        prob.solve()

        # Solve the thirtiary objective of ensuring that devices are turned on as early as possible.
        prob += z == opt_value, "FixMainObjective2"
        prob.setObjective(
            lpSum(
                t * schedule[device.device_name, t]
                for t in range(number_timesteps)
                for device in self.socket_and_battery_list
            )
        )
        prob.solve()

        # Output results
        print("Status:", LpStatus[prob.status])
        print("Max imbalance z:", value(z))
        for t in range(number_timesteps):
            print(f"Time {t}: Power imbalance = {value(P_l[t])}")
            actual = P_p[t] - sum(
                (value(schedule.get((device.device_name, t), 0.0)) or 0.0)
                * device.max_power_usage
                for device in self.socket_and_battery_list
            )
            print(
                f"t={t}: P_l={value(P_l[t]):.3f}, actual={actual:.3f}, diff={abs(value(P_l[t]) - actual):.6f}"
            )
        for device in self.socket_and_battery_list:
            df_power_interpolated[f"schedule {device.device_name}"] = 0.0
            for t, datetime_index in enumerate(df_power_interpolated.index):
                print(
                    f"Device {device.device_name} at time {t}: O = {value(schedule[device.device_name, t])}, E_s = {value(E_s[device.device_name, t])}"
                )
                df_power_interpolated.at[
                    datetime_index, f"schedule {device.device_name}"
                ] = value(schedule[device.device_name, t])
        print("Optimization finished")
        return df_power_interpolated


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
