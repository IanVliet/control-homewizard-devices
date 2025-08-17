import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import time

# from control_homewizard_devices.schedule_devices_pulp import (
#     DeviceSchedulingOptimizationPULP,
# )
# from control_homewizard_devices.schedule_devices_scip import (
#     DeviceSchedulingOptimizationSCIP,
#     print_schedule_results,
# )
from control_homewizard_devices.schedule_devices import (
    DeviceSchedulingOptimization,
    print_schedule_results,
)
from control_homewizard_devices.device_classes import SocketDevice, Battery
from pathlib import Path
import argparse

DELTA_T_BENCHMARK = 0.25  # 15 minutes in hours

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug-scheduler",
    action="store_true",
    help="Enable debug output for the scheduler related tests",
)
parser.add_argument(
    "--single-schedule",
    nargs="?",
    const=10,
    type=int,
    default=None,
    help="Only obtain a single schedule with the indicated size. Uses the value provided or a default value of 10 if no value is provided.",
)
args = parser.parse_args()


def get_power_dataframe(size: int, power_kw=1) -> pd.DataFrame:
    data = np.zeros(size)

    half = size // 2
    quarter = size // 4
    start = half - quarter
    end = half + quarter
    data[start:end] = power_kw

    start_timestamp = pd.Timestamp("2025-01-01 08:00:00").replace(
        tzinfo=ZoneInfo("Europe/Amsterdam")
    )
    index = pd.date_range(
        start=start_timestamp, periods=size, freq=f"{DELTA_T_BENCHMARK}h"
    )

    df_power = pd.DataFrame(data, index=index, columns=["power_kw"])
    return df_power


def get_solar_prediction_dataframe(size: int) -> pd.DataFrame:
    base_dir = Path(__file__).parent.parent.parent
    df_solar_prediction = pd.read_parquet(
        base_dir / "data" / "solar_prediction.parquet"
    )
    df_solar_prediction.index = pd.to_datetime(df_solar_prediction.index)
    date_0800 = (
        df_solar_prediction.index[0] + pd.Timedelta(days=1)
    ).normalize() + pd.Timedelta(hours=8)
    start_index = df_solar_prediction.index.get_indexer([date_0800], method="nearest")[
        0
    ]
    df_solar_prediction = df_solar_prediction.iloc[start_index : start_index + size]
    return df_solar_prediction


def run_benchmark():
    """
    Run the scheduling benchmark for HomeWizard devices.
    """
    size = 4
    if args.single_schedule is not None:
        size = args.single_schedule
    data_multiplier = 2
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket kitchen boiler",
            2000,
            2000,
            1,
            True,
            delta_t=DELTA_T_BENCHMARK,
        ),
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket shower boiler",
            1000,
            7000,
            2,
            False,
            delta_t=DELTA_T_BENCHMARK,
        ),
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            3000,
            3,
            {"name": "test_user", "token": ""},
        ),
    ]

    while True:
        # power_df = get_power_dataframe(size)
        power_df = get_solar_prediction_dataframe(size)
        start = time.time()
        optimization = DeviceSchedulingOptimization(DELTA_T_BENCHMARK)
        data, results = optimization.solve_schedule_devices(
            power_df, devices_list, time_limit=10
        )
        end = time.time()
        scheduling_time = end - start
        print(f"Scheduling with size {size} took {scheduling_time:3f} seconds")
        if scheduling_time > 10:
            print(f"Scheduling took too long, max data size {size}.")
            if args.debug_scheduler:
                print_schedule_results(data, results)
            break
        if args.single_schedule is not None:
            print(
                f"single-schedule flag used, stopping benchmark with data size {args.single_schedule}"
            )
            break
        size *= data_multiplier
        if size > 200:
            print("Reached maximum data size limit, stopping benchmark.")
            break


if __name__ == "__main__":
    run_benchmark()
