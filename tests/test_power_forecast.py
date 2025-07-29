import pytest
import pandas as pd
from control_homewizard_devices.device_classes import SocketDevice
from control_homewizard_devices.power_forecast import schedule_devices


@pytest.fixture(scope="session")
def solar_prediction_data():
    df_solar_prediction = pd.read_parquet("../data/solar_prediction.parquet")


def test_all_scheduled_on():
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=3, freq="15min")
    df_high_power = pd.DataFrame({"power_kw": [1, 1.5, 2]}, index=datetimes)
    devices_list = [SocketDevice("", "HWE-SKT", "test socket", 1000, 750, 1, True)]
    df_schedules = schedule_devices(df_high_power, devices_list, 900)
    print(
        df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [1, 1, 1]
    )
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
    ]


def test_schedule_first_device_on_first_half_second_device_second_half():
    """
    The first device with small capacity should be turned on for the first half of the prediction.
    The second device with larger capacity and higher priority should be turned on when the predicted available power is higher.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes)
    devices_list = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 2, False),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 500, 1, True),
    ]
    df_schedules = schedule_devices(df_predicted_power, devices_list, 900)
    assert (
        df_schedules[f"schedule {devices_list[0].device_name}"].to_list()
        == [1, 1, 0, 0]
    ) and (
        df_schedules[f"schedule {devices_list[1].device_name}"].to_list()
        == [0, 0, 1, 1]
    )


def test_schedule_second_device_only():
    """
    The first device with small capacity not be turned on at all,
    due to the second device with larger capacity and
    higher priority needing to be turned on during the entire duration.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes)
    devices_list = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 2, False),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 1000, 1, True),
    ]
    df_schedules = schedule_devices(df_predicted_power, devices_list, 900)
    assert (
        df_schedules[f"schedule {devices_list[0].device_name}"].to_list()
        == [0, 0, 0, 0]
    ) and (
        df_schedules[f"schedule {devices_list[1].device_name}"].to_list()
        == [1, 1, 1, 1]
    )


def test_schedule_battery_charge_second_device_charge():
    """
    First the battery should be turned on until enough energy is available such that second device can be charged.
    """
    # TODO: Create battery class
    assert False
