from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime
import json
from plotly import graph_objects as go
import pandas as pd
from control_homewizard_devices.device_classes import SocketDevice
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpInteger,
    LpStatus,
    value,
)


def schedule_devices(
    df_power: pd.DataFrame, socket_list: list[SocketDevice], delta_t_sec: int = 30
):
    new_datetime_index = pd.date_range(
        start=df_power.index[0], end=df_power.index[-1], freq=f"{delta_t_sec}s"
    )
    delta_t = delta_t_sec / 3600  # convert to hours such that the energy is in kWh
    for socket in socket_list:
        socket.energy_stored = 0

    # create the problem to be optimized
    prob = LpProblem("Device_Scheduling_Optimization", LpMinimize)

    df_power_interpolated = df_power.reindex(new_datetime_index).interpolate(
        method="time"
    )
    P_p = (df_power_interpolated["power_kw"] * 1000).to_list()
    number_timesteps = len(P_p)

    # Variables
    schedule = {
        (socket.device_name, t): LpVariable(
            f"O_{socket.device_name}_{t}", -1, 1, cat=LpInteger
        )
        for socket in socket_list
        for t in range(number_timesteps)
    }
    E_s = {
        (socket.device_name, t): LpVariable(
            f"E_s_{socket.device_name}_{t}", 0, socket.energy_capacity
        )
        for socket in socket_list
        for t in range(number_timesteps)
    }  # Energy stored
    P_l = {
        t: LpVariable(f"P_l_{t}") for t in range(number_timesteps)
    }  # Power left over
    z = LpVariable("z")

    # Objective: minimize max absolute power imbalance
    prob += z

    # Constraints
    for t in range(number_timesteps):
        # Power left
        prob += P_l[t] == P_p[t] - lpSum(
            schedule[socket.device_name, t] * socket.max_power_usage
            for socket in socket_list
        )
        prob += P_l[t] <= z
        prob += -P_l[t] <= z

    for socket in socket_list:
        for t in range(number_timesteps):
            if t == 0:
                prob += (
                    E_s[socket.device_name, t]
                    == socket.energy_stored
                    + schedule[socket.device_name, t] * socket.max_power_usage * delta_t
                )
            else:
                prob += (
                    E_s[socket.device_name, t]
                    == E_s[socket.device_name, t - 1]
                    + schedule[socket.device_name, t] * socket.max_power_usage * delta_t
                )

            prob += E_s[socket.device_name, t] <= socket.energy_capacity
            prob += E_s[socket.device_name, t] >= 0

            # Non-needed devices can't charge when power is negative
            # if not socket.daily_need:
            #     prob += schedule[socket.device_name, t] >= 0  # no discharging

        # Final energy requirement
        if socket.daily_need:
            prob += (
                E_s[socket.device_name, number_timesteps - 1] >= socket.energy_capacity
            )

    # Device control limits
    for socket in socket_list:
        for t in range(number_timesteps):
            prob += schedule[socket.device_name, t] >= 0  # only 0 or 1

    # Solve the problem
    prob.solve()

    # Output results
    print("Status:", LpStatus[prob.status])
    print("Max imbalance z:", value(z))
    for t in range(number_timesteps):
        print(f"Time {t}: Power imbalance = {value(P_l[t])}")
        actual = P_p[t] - sum(
            (value(schedule.get((socket.device_name, t), 0.0)) or 0.0)
            * socket.max_power_usage
            for socket in socket_list
        )
        print(
            f"t={t}: P_l={value(P_l[t]):.3f}, actual={actual:.3f}, diff={abs(value(P_l[t]) - actual):.6f}"
        )
    for socket in socket_list:
        df_power_interpolated[f"schedule {socket.device_name}"] = 0
        for t, datetime_index in enumerate(df_power_interpolated.index):
            print(
                f"Device {socket.device_name} at time {t}: O = {value(schedule[socket.device_name, t])}, E_s = {value(E_s[socket.device_name, t])}"
            )
            df_power_interpolated.at[
                datetime_index, f"schedule {socket.device_name}"
            ] = value(schedule[socket.device_name, t])
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
