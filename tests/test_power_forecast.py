import pytest
import pandas as pd
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    Battery,
)
from control_homewizard_devices.power_forecast import schedule_devices


@pytest.fixture(scope="session")
def solar_prediction_data():
    df_solar_prediction = pd.read_parquet("../data/solar_prediction.parquet")


def test_single_device_only_on_needed():
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=3, freq="15min")
    df_high_power = pd.DataFrame({"power_kw": [1, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 750, 1, True)
    ]
    df_schedules = schedule_devices(df_high_power, devices_list, 900)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
    ]


def test_single_device_only_on_optional():
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=3, freq="15min")
    df_high_power = pd.DataFrame({"power_kw": [1, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 750, 1, False)
    ]
    df_schedules = schedule_devices(df_high_power, devices_list, 900)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
    ]


def test_single_device_on_and_off():
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_high_power = pd.DataFrame({"power_kw": [1, 1, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 500, 1, True)
    ]
    df_schedules = schedule_devices(df_high_power, devices_list, 900)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


def test_schedule_first_device_on_first_half_second_device_second_half():
    """
    The first device with small capacity should be turned on for the first half of the prediction.
    The second device with larger capacity and higher priority should be turned on when the predicted available power is higher.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
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
    The first device with small capacity should not be turned on at all,
    due to the second device with larger capacity and
    higher priority should be turned on the entire duration.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
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
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame(
        {"power_kw": [0.5, 0.5, 0.5, 0.5]}, index=datetimes
    )
    test_socket = SocketDevice(
        "", "HWE-SKT", "test socket high power", 1000, 500, 1, True
    )
    test_battery = Battery(
        "", "HWE-BAT", "test battery", 500, 2000, {"name": "test_user", "token": ""}
    )
    socket_and_battery_list: list[SocketDevice | Battery] = [test_socket, test_battery]
    df_schedules = schedule_devices(df_predicted_power, socket_and_battery_list, 900)
    # expects switching between states to occur as little as possible (so 0, 0, 1, 1 instead of 0, 1, 0, 1)
    assert (
        df_schedules[f"schedule {socket_and_battery_list[0].device_name}"].to_list()
        == [0, 0, 1, 1]
    ) and (
        df_schedules[f"schedule {socket_and_battery_list[1].device_name}"].to_list()
        == [1, 1, -1, -1]
    )


# TODO: Create test for checking if load is spread out when multiple devices are needed.
def test_load_balance():
    """
    Although both devices are needed, the devices should not be on at the same time.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    df_predicted_power = pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 1, True),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 500, 2, True),
    ]
    # TODO: Optionally make schedule devices be influenced by priority... (Sort devices by priority?)
    df_schedules = schedule_devices(df_predicted_power, devices_list, 900)
    assert (
        df_schedules[f"schedule {devices_list[0].device_name}"].to_list()
        == [1, 1, 0, 0]
    ) and (
        df_schedules[f"schedule {devices_list[1].device_name}"].to_list()
        == [0, 0, 1, 1]
    )


def test_only_charge_battery():
    """
    The battery should be expected to charge until full, but not discharge unnessarily.
    """
    datetimes = pd.date_range(start="2025-01-01 09:00", periods=4, freq="15min")
    # TODO: Fix schedule devices to actually also turn on when there is enough power even though the power is the same at each moment.
    # Also why if it is descending is it in the right order, and if it ascending power it is in the opposite order?
    df_predicted_power = pd.DataFrame({"power_kw": [1, 1.2, 1.2, 1.3]}, index=datetimes)
    devices_list: list[SocketDevice | Battery] = [
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            600,
            {"name": "test_user", "token": ""},
        )
    ]
    df_schedules = schedule_devices(df_predicted_power, devices_list, 900)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# TODO: Create test for checking if maximum capacity of batteries is taken into account properly.

# TODO: Create test for checking if maximum capacity of sockets is taken into account properly.

# TODO: Implement logic to substract a constant from the prediced power based on the difference between predicted power from solar
# and the actual available power at the current moment due to e.g. other devices or inaccurate prediction of available solar.
