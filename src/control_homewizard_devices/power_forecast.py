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


def schedule_devices(
    df_power: pd.DataFrame,
    socket_and_battery_list: list[SocketDevice | Battery],
    delta_t_sec: int = 30,
):
    new_datetime_index = pd.date_range(
        start=df_power.index[0], end=df_power.index[-1], freq=f"{delta_t_sec}s"
    )
    delta_t = delta_t_sec / 3600  # convert to hours such that the energy is in kWh

    # create the problem to be optimized
    prob = LpProblem("Device_Scheduling_Optimization", LpMinimize)

    df_power_interpolated = df_power.reindex(new_datetime_index).interpolate(
        method="time"
    )
    P_p = (df_power_interpolated["power_kw"] * 1000).to_list()
    number_timesteps = len(P_p)

    # Variables (with their limits)
    # in schedule: a device with -1 provides power (e.g. a battery discharging), 0 a device does nothing, 1 a device consumes power
    # TODO: The batteries can charge and discharge up to their max power, but can also do in between, therefore we should turn LpInteger into LpContinuous.
    schedule = {
        (device.device_name, t): LpVariable(
            f"O_{device.device_name}_{t}",
            -1,
            1,
            cat=LpContinuous if isinstance(device, Battery) else LpInteger,
        )
        for device in socket_and_battery_list
        for t in range(number_timesteps)
    }
    # Energy stored
    E_s = {
        (device.device_name, t): LpVariable(
            f"E_s_{device.device_name}_{t}", 0, device.energy_capacity
        )
        for device in socket_and_battery_list
        for t in range(number_timesteps)
    }
    # power available variable
    P_l = {t: LpVariable(f"P_l_{t}") for t in range(number_timesteps)}
    # combined objective
    z = LpVariable("z")

    # Objective: minimize max absolute power imbalance
    # since this is the first time we would use the prob += syntax, we can set the objective with prob += z.
    prob += z

    # Constraints (after the first prob +=, the prob += syntax is used to add constraints)
    for t in range(number_timesteps):
        # Power left
        prob += P_l[t] == P_p[t] - lpSum(
            schedule[device.device_name, t] * device.max_power_usage
            for device in socket_and_battery_list
        )
        prob += P_l[t] <= z
        prob += -P_l[t] <= z

    for device in socket_and_battery_list:
        for t in range(number_timesteps):
            if t == 0:
                prob += (
                    E_s[device.device_name, t]
                    == device.energy_stored
                    + schedule[device.device_name, t] * device.max_power_usage * delta_t
                )
            else:
                prob += (
                    E_s[device.device_name, t]
                    == E_s[device.device_name, t - 1]
                    + schedule[device.device_name, t] * device.max_power_usage * delta_t
                )

            prob += E_s[device.device_name, t] <= device.energy_capacity
            prob += E_s[device.device_name, t] >= 0

            # Non-needed devices can't charge when power is negative
            # if not socket.daily_need:
            #     prob += schedule[socket.device_name, t] >= 0  # no discharging

        # Final energy requirement
        if isinstance(device, SocketDevice) and device.daily_need:
            prob += (
                E_s[device.device_name, number_timesteps - 1] >= device.energy_capacity
            )

    # Device control limits (sockets can only consume power or do nothing)
    for device in socket_and_battery_list:
        if isinstance(device, SocketDevice):
            for t in range(number_timesteps):
                prob += schedule[device.device_name, t] >= 0  # only 0 or 1

    # Solve the problem
    prob.solve()

    # Solve the secondary objective of minimizing the number of times sockets are switched on/off
    opt_value = value(prob.objective)
    switches = {
        (device.device_name, t): LpVariable(
            f"switch_{device.device_name}_{t}", 0, 1, cat=LpBinary
        )
        for device in socket_and_battery_list
        for t in range(number_timesteps - 1)
    }
    diff = {
        (device.device_name, t): LpVariable(
            f"diff_{device.device_name}_{t}", -2, 2, cat=LpInteger
        )
        for device in socket_and_battery_list
        for t in range(number_timesteps - 1)
    }
    prob += z == opt_value, "FixMainObjective"

    for device in socket_and_battery_list:
        for t in range(number_timesteps - 1):
            prob += (
                diff[device.device_name, t]
                == schedule[device.device_name, t + 1] - schedule[device.device_name, t]
            )
            prob += diff[device.device_name, t] <= 2 * switches[device.device_name, t]
            prob += -diff[device.device_name, t] <= 2 * switches[device.device_name, t]

    prob.setObjective(
        lpSum(
            switches[device.device_name, t]
            for t in range(number_timesteps - 1)
            for device in socket_and_battery_list
        )
    )
    prob.solve()

    # Solve the thirtiary objective of ensuring that devices are turned on as early as possible.
    prob += z == opt_value, "FixMainObjective2"
    prob.setObjective(
        lpSum(
            t * schedule[device.device_name, t]
            for device in socket_and_battery_list
            for t in range(number_timesteps)
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
            for device in socket_and_battery_list
        )
        print(
            f"t={t}: P_l={value(P_l[t]):.3f}, actual={actual:.3f}, diff={abs(value(P_l[t]) - actual):.6f}"
        )
    for device in socket_and_battery_list:
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
