import pytest
import pandas as pd
from control_homewizard_devices.device_classes import (
    SocketDevice,
    Battery,
)
from control_homewizard_devices.schedule_devices import (
    DeviceSchedulingOptimization,
    print_schedule_results,
)
from control_homewizard_devices.constants import AGGREGATE_BATTERY

DELTA_T_TEST = 0.25  # 15 minutes in hours


@pytest.fixture(scope="module")
def datetimes_delta_t():
    return pd.date_range(
        start="2025-01-01 09:00",
        periods=4,
        freq=f"{DELTA_T_TEST}h",
    )


@pytest.fixture(scope="module")
def power_1kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [1] * 4}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_0_5kw_and_1kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [0.5, 0.5, 1, 1]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_ascending(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [0.5, 0.75, 1, 1.25]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_descending(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [1.25, 1, 0.75, 0.5]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_1kw_neg_1kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [1, 1, -1, -1]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def power_1kw_neg_1_5kw_pos_2kw(datetimes_delta_t):
    return pd.DataFrame({"power_kw": [1, 1, -1.5, 2]}, index=datetimes_delta_t)


@pytest.fixture(scope="module")
def datetimes_2_delta_t():
    return pd.date_range(
        start="2025-01-01 09:00",
        periods=4,
        freq=f"{2 * DELTA_T_TEST}h",
    )


@pytest.fixture(scope="module")
def power_1kw_2_delta_t(datetimes_2_delta_t):
    return pd.DataFrame({"power_kw": [1, 1, 1, 1]}, index=datetimes_2_delta_t)


# Tests to ensure a device is turned on when there is enough power.
def test_single_device_only_on_needed(power_1kw, request):
    """
    All the devices should be scheduled to turn on, because there is enough power.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 1000, 1, True, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)

    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


def test_single_socket_only_on_optional(power_1kw, request):
    """
    The socket should be scheduled to turn on for the entire duration,
    because there is enough power and the socket can store all energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 1000, 1, False, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)

    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


def test_single_battery_only_on(power_1kw, request):
    """
    The battery should be scheduled to turn on for the entire duration,
    because there is enough power and the battery can store all energy.
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
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)

    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)

    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)

    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


# Tests to ensure devices are turned on as early as possible.
def test_single_device_on_and_off(power_1kw, request):
    """
    The socket should be scheduled to turn on at the beginning but not at the end,
    because there is enough power available, but the device cannot store all energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 500, 1, True, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)

    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


def test_charge_battery_until_full(power_1kw, request):
    """
    The battery should be expected to charge until full as quickly as possible,
    but not discharge unnessarily.
    """
    # Also why if it is descending is it in the right order,
    # and if it ascending power it is in the opposite order?
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
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)

    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# Test to ensure a needed device is turned on before an optional device is turned on
# even when this would require power from the grid.
def test_schedule_second_device_only(power_0_5kw_and_1kw, request):
    """
    The first device with a smaller capacity should not be turned on at all,
    since the second device with a larger capacity and
    higher priority should be turned on for the entire duration.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket low power",
            500,
            250,
            2,
            False,
            delta_t=DELTA_T_TEST,
        ),
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket high power",
            1000,
            1000,
            1,
            True,
            delta_t=DELTA_T_TEST,
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_0_5kw_and_1kw, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
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


# Test to ensure a needed device is turned on later
# if the device can still be fully charged.
def test_sufficient_power_for_spread_out_activation_one_optional_one_needed(
    power_0_5kw_and_1kw, request
):
    """
    The first device with small capacity should be turned on
    for the first half of the prediction.
    The second device with larger capacity and higher priority should be turned on
    when the predicted available power is higher.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket low power",
            500,
            250,
            2,
            False,
            delta_t=DELTA_T_TEST,
        ),
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket high power",
            1000,
            500,
            1,
            True,
            delta_t=DELTA_T_TEST,
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_0_5kw_and_1kw, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
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


# Test for checking if load is spread out when multiple devices are needed,
# but enough power is available.
def test_sufficient_power_for_spread_out_activation_two_needed_devices(
    power_0_5kw_and_1kw, request
):
    """
    Although both devices are needed, the devices should not be on at the same time
    since they can be on after each other.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket low power",
            500,
            250,
            1,
            True,
            delta_t=DELTA_T_TEST,
        ),
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket high power",
            1000,
            500,
            2,
            True,
            delta_t=DELTA_T_TEST,
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_0_5kw_and_1kw, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
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


# Test to ensure a battery is charged first before a second device is turned on
# if the device would have to use power from the grid otherwise.
def test_schedule_battery_charge_second_device_charge(power_1kw, request):
    """
    First the battery should be turned on until enough energy is available
    such that second device can be charged.
    """
    test_socket = SocketDevice(
        "",
        "HWE-SKT",
        "test socket high power",
        2000,
        1000,
        1,
        True,
        delta_t=DELTA_T_TEST,
    )
    test_battery = Battery(
        "",
        "HWE-BAT",
        "test battery",
        1000,
        4000,
        {"name": "test_user", "token": ""},
    )
    devices_list: list[SocketDevice | Battery] = [test_socket, test_battery]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    # expects switching between states to occur as little as possible
    # (so 0, 0, 1, 1 instead of 0, 1, 0, 1 for the socket)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        0,
        0,
        1,
        1,
    ]
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == [
        1,
        1,
        -1,
        -1,
    ]


# Test if load is still spread out
# when there is not enough power available for both devices.
def test_insufficient_power_for_activation_two_needed_devices(
    power_0_5kw_and_1kw, request
):
    """
    Although both devices are needed, the devices should not be on at the same time
    since there is not enough power.
    If it does not influence the power balance
    the devices should be in the same state as long as possible
    (on until full, then off).
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "1st test socket", 1000, 500, 1, True, delta_t=DELTA_T_TEST
        ),
        SocketDevice(
            "", "HWE-SKT", "2nd test socket", 1000, 500, 2, True, delta_t=DELTA_T_TEST
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_0_5kw_and_1kw, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        0,
        0,
        1,
        1,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# Tests for checking if maximum capacity of devices is taken into account properly.
def test_maximum_capacity_of_battery(power_1kw, request):
    """
    The battery should not be scheduled to turn on for the entire duration,
    because it cannot store all energy.
    However, it should charge to its maximum capacity by using a ratio in the schedule.
    (In practice the battery will use full power until it reaches its maximum capacity.)
    """
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
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == pytest.approx(
        [
            1,
            1,
            0.4,
            0,
        ]
    )


def test_maximum_storage_of_sockets(power_1kw, request):
    """
    The socket should not be scheduled to turn on for the entire duration,
    because it cannot store all energy.
    However, it should charge to its maximum capacity
    by overcharging for one extra step.
    (In practice the device will stop consuming power
    when it reaches its maximum capacity.)
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket",
            1000,
            600,
            1,
            True,
            delta_t=DELTA_T_TEST,
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        0,
    ]


# Test for cases of not perfectly matching available power
# with the power usage of the devices.
def test_not_perfectly_matching_power(power_1kw, request):
    """
    The socket should only be turned on for the first part
    of the prediction, because it cannot store all the available energy.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 900, 450, 1, True, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# Test for cases where an optional device cannot be fully charged,
# but is still charged as much as possible.
def test_optional_device_not_fully_charged(power_descending, request):
    """
    The socket should be scheduled to turn on for the first two steps only,
    resulting in the device not being fully charged,
    because there is not enough power available in later stages.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 1000, 1, False, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_descending, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


def test_devices_order_despite_power(power_descending, request):
    """
    The socket with highest priority (smallest number) should be scheduled
    to turn on for the first two steps
    and the second socket for first and third step
    (since these have the highest available power remaining),
    this does not result in the least amount of power going to and from the grid.
    (The algorithm is not guaranteed to find the optimal solution,
    but it should be close.)
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 500, 2, True, delta_t=DELTA_T_TEST
        ),
        SocketDevice(
            "", "HWE-SKT", "test socket 2", 500, 250, 1, True, delta_t=DELTA_T_TEST
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_descending, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        0,
        1,
        0,
    ]
    assert df_schedules[f"schedule {devices_list[1].device_name}"].to_list() == [
        1,
        1,
        0,
        0,
    ]


# Test for a case where a needed device cannot be fully charged,
# but is still charged as much as possible.
# Without resulting in the problem being infeasible.
def test_needed_device_not_fully_charged(power_ascending, request):
    """
    The socket should be scheduled to turn on for the first two steps only,
    resulting in the device not being fully charged,
    because there is not enough power available in later stages.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 2000, 1, True, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_ascending, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        1,
    ]


# test which considers the use of interpolation
# (so different delta_t for schedule devices than for the power forecast).
def test_device_on_until_full_different_delta_t(power_1kw_2_delta_t, request):
    """
    The socket should be scheduled to turn on for the first three steps only,
    resulting in the device being perfectly charged,
    because there is enough power available and
    the timestep of delta_t_test/2 doubles the number of timesteps.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 750, 1, True, delta_t=DELTA_T_TEST
        )
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_1kw_2_delta_t, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    # print(power_1kw_2_delta_t)
    # print(df_schedules)
    assert df_schedules[f"schedule {devices_list[0].device_name}"].to_list() == [
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]


def test_battery_discharge(power_1kw_neg_1kw, request):
    """
    The battery should be scheduled to turn on for the entire duration,
    because it can discharge all energy.
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
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(power_1kw_neg_1kw, devices_list)
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == [
        1,
        1,
        -1,
        -1,
    ]


def test_battery_discharge_with_socket(power_1kw_neg_1_5kw_pos_2kw, request):
    """
    The battery should first charge then discharge,
    at the end there should be enough energy left to charge the socket.
    """
    devices_list: list[SocketDevice | Battery] = [
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            1000,
            {"name": "test_user", "token": ""},
        ),
        SocketDevice(
            "", "HWE-SKT", "test socket", 3000, 750, 2, True, delta_t=DELTA_T_TEST
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_1kw_neg_1_5kw_pos_2kw, devices_list
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == pytest.approx(
        [
            1,
            1,
            -1,
            -1,
        ]
    )
    assert df_schedules[
        f"schedule {devices_list[1].device_name}"
    ].to_list() == pytest.approx(
        [
            0,
            0,
            0,
            1,
        ]
    )


# Create test where the socket is on even after the device is full
# Only if enough power is available anyways.
# This should only happen with the overcharge option set to True
def test_socket_on_overcharge(power_1kw, request):
    """
    If enough power is available, turn socket on,
    even if device is already fully charged
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "", "HWE-SKT", "test socket", 1000, 500, 1, True, delta_t=DELTA_T_TEST
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_1kw, devices_list, overcharge=True
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[
        f"schedule {devices_list[0].device_name}"
    ].to_list() == pytest.approx(
        [
            1,
            1,
            1,
            1,
        ]
    )


def test_socket_on_minimum_battery_charge(power_1kw, request):
    """
    An optional device with a minimum charge parameter turns on
    when the batteries have atleast a certain percentage of charge.
    Before that the device does not turn on.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket",
            1000,
            500,
            1,
            False,
            delta_t=DELTA_T_TEST,
            min_battery_charge=0.5,
        ),
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            1000,
            {"name": "test_user", "token": ""},
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_1kw, devices_list, overcharge=True
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[
        f"schedule {devices_list[0].device_name}"
    ].to_list() == pytest.approx(
        [
            0,
            0,
            1,
            1,
        ]
    )
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == pytest.approx(
        [
            1,
            1,
            0,
            0,
        ]
    )


def test_multiple_socket_on_minimum_battery_charge(power_1kw, request):
    """
    An optional device with a minimum charge parameter turns on
    when the batteries have atleast a certain ratio of charge.
    This also ensures devices with lower priority turn on later.
    Before that both devices do not turn on.
    """
    devices_list: list[SocketDevice | Battery] = [
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket 1",
            1000,
            250,
            1,
            False,
            delta_t=DELTA_T_TEST,
            min_battery_charge=0.5,
        ),
        SocketDevice(
            "",
            "HWE-SKT",
            "test socket 2",
            1000,
            500,
            2,
            False,
            delta_t=DELTA_T_TEST,
            min_battery_charge=0.0,
        ),
        Battery(
            "",
            "HWE-BAT",
            "test battery",
            1000,
            1000,
            {"name": "test_user", "token": ""},
        ),
    ]
    optimization = DeviceSchedulingOptimization(DELTA_T_TEST)
    data, results = optimization.solve_schedule_devices(
        power_1kw, devices_list, overcharge=True
    )
    if request.config.getoption("--debug-scheduler"):
        print_schedule_results(data, results)
    df_schedules = results[-1].df_variables
    assert df_schedules[
        f"schedule {devices_list[0].device_name}"
    ].to_list() == pytest.approx(
        [
            0,
            0,
            1,
            0,
        ]
    )
    assert df_schedules[
        f"schedule {devices_list[1].device_name}"
    ].to_list() == pytest.approx(
        [
            0,
            0,
            0,
            1,
        ]
    )
    assert df_schedules[f"schedule {AGGREGATE_BATTERY}"].to_list() == pytest.approx(
        [
            1,
            1,
            0,
            0,
        ]
    )
