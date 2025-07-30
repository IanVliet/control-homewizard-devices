import pytest
import pandas as pd
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    Battery,
)
from control_homewizard_devices.power_forecast import schedule_devices
import control_homewizard_devices.device_classes as device_classes

DELTA_T_TEST = 0.25  # 15 minutes in hours


@pytest.fixture(scope="session")
def solar_prediction_data():
    df_solar_prediction = pd.read_parquet("../data/solar_prediction.parquet")


@pytest.fixture(autouse=True)
def patch_delta_t(monkeypatch):
    monkeypatch.setattr(
        device_classes,
        "DELTA_T",
        DELTA_T_TEST,
    )


def test_override_delta_t_15min():
    assert device_classes.DELTA_T == 0.25


@pytest.fixture(scope="module")
def datetimes_delta_t():
    return pd.date_range(
        start="2025-01-01 09:00",
        periods=4,
        freq=f"{DELTA_T_TEST}h",
    )


@pytest.fixture(scope="module")
def power_1kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [1, 1, 1, 1]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_0_5kw_and_1kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes_delta_t)


# Tests to ensure a device is turned on when there is enough power.
def test_single_device_only_on_needed(power_1kw):
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 1000, 1, True)
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


def test_single_socket_only_on_optional(power_1kw):
    """
    The socket should be scheduled to turn on for the entire duration, because there is enough power and the socket can store all energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 1000, 1, False)
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


def test_single_battery_only_on(power_1kw):
    """
    The battery should be scheduled to turn on for the entire duration, because there is enough power and the battery can store all energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            1000,
            {"name": "test_user", "token": ""},
        )
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


# Tests to ensure devices are turned on as early as possible.
def test_single_device_on_and_off(power_1kw):
    """
    The socket should be scheduled to turn on at the beginning but not at the end, because there is enough power available, but the device cannot store all energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 500, 1, True)
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


def test_charge_battery_until_full(power_1kw):
    """
    The battery should be expected to charge until full as quickly as possible, but not discharge unnessarily.
    """
    # Also why if it is descending is it in the right order, and if it ascending power it is in the opposite order?
    devices_list: list[SocketDevice | Battery] = [
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            500,
            {"name": "test_user", "token": ""},
        )
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# Test to ensure a needed device is turned on before an optional device is turned on even when this would require power from the grid.
def test_schedule_second_device_only(power_0_5kw_and_1kw):
    """
    The first device with a smaller capacity should not be turned on at all,
    since the second device with a larger capacity and
    higher priority should be turned on for the entire duration.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 2, False),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 1000, 1, True),
    ]
    df_schedules = schedule_devices(
        power_0_5kw_and_1kw, devices_list, device_classes.DELTA_T
    )
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        0,
        0,
        0,
        0,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


# Test to ensure a needed device is turned on later if the device can still be fully charged.
def test_sufficient_power_for_spread_out_activation_one_optional_one_needed(
    power_0_5kw_and_1kw,
):
    """
    The first device with small capacity should be turned on for the first half of the prediction.
    The second device with larger capacity and higher priority should be turned on when the predicted available power is higher.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 2, False),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 500, 1, True),
    ]
    df_schedules = schedule_devices(
        power_0_5kw_and_1kw, devices_list, device_classes.DELTA_T
    )
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        0,
        0,
        1,
        1,
    ]


# Test for checking if load is spread out when multiple devices are needed, but enough power is available.
def test_sufficient_power_for_spread_out_activation_two_needed_devices(
    power_0_5kw_and_1kw,
):
    """
    Although both devices are needed, the devices should not be on at the same time since they can be on after each other.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket low power", 500, 250, 1, True),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 500, 2, True),
    ]
    # TODO: Optionally make schedule devices be influenced by priority... (Sort devices by priority?)
    df_schedules = schedule_devices(
        power_0_5kw_and_1kw, devices_list, device_classes.DELTA_T
    )
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        0,
        0,
        1,
        1,
    ]


# Test to ensure a battery is charged first before a second device is turned on if the device would have to use power from the grid otherwise.
def test_schedule_battery_charge_second_device_charge(power_1kw):
    """
    First the battery should be turned on until enough energy is available such that second device can be charged.
    """
    test_socket = SocketDevice(
        "", "HWE-SKT", "test socket high power", 2000, 1000, 1, True
    )
    test_battery = Battery(
        "", "HWE-BAT", "test battery", 1000, 4000, {"name": "test_user", "token": ""}
    )
    socket_and_battery_list: list[SocketDevice | Battery] = [test_socket, test_battery]
    df_schedules = schedule_devices(
        power_1kw, socket_and_battery_list, device_classes.DELTA_T
    )
    # expects switching between states to occur as little as possible (so 0, 0, 1, 1 instead of 0, 1, 0, 1 for the socket)
    assert df_schedules[
        f"schedule {socket_and_battery_list[0].device_name}"
    ].to_list() == [0, 0, 1, 1]
    assert df_schedules[
        f"schedule {socket_and_battery_list[1].device_name}"
    ].to_list() == [1, 1, -1, -1]


# Test if load is still spread out when there is not enough power available for both devices.
def test_insufficient_power_for_activation_two_needed_devices(power_0_5kw_and_1kw):
    """
    Although both devices are needed, the devices should not be on at the same time since there is not enough power.
    However, the devices should still not activate at the same time.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket low power", 1000, 500, 1, True),
        SocketDevice("", "HWE-SKT", "test socket high power", 1000, 500, 2, True),
    ]
    df_schedules = schedule_devices(
        power_0_5kw_and_1kw, devices_list, device_classes.DELTA_T
    )
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        0,
        0,
        1,
        1,
    ]


# TODO: Create test for checking if maximum capacity of batteries is taken into account properly.


# TODO: Create test for checking if maximum capacity of sockets is taken into account properly.
def test_maximum_capacity_of_sockets(power_1kw):
    """
    The socket should not be scheduled to turn on for the entire duration, because it cannot store all energy.
    However, it should charge to its maximum capacity. (Since in practice the device will stop consuming power when it reaches its maximum capacity.)
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice("", "HWE-SKT", "test socket", 1000, 600, 1, True)
    ]
    df_schedules = schedule_devices(power_1kw, devices_list, device_classes.DELTA_T)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        0,
    ]


# TODO: Create test for cases of not perfectly matching available power with the capacity of the devices.

# TODO: Create test for cases where an optional device cannot be fully charged, but is still charged as much as possible.

# TODO: Create test for cases where a needed device cannot be fully charged, but is still charged as much as possible.

# TODO: Create tests which consider the use of interpolation (so different delta_t for schedule devices than for the power forecast).

# TODO: Implement logic to substract a constant from the prediced power based on the difference between predicted power from solar
# and the actual available power at the current moment due to e.g. other devices or inaccurate prediction of available solar.
