import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from control_homewizard_devices.device_classes import Battery, MarstekBattery
from control_homewizard_devices.marstek_api.udp_request import (
    InvalidMarstekResponseError,
    MarstekESStatus,
    parse_marstek_es_status,
)
from control_homewizard_devices.utils import DEVICE_TYPE_MAP


def test_marstek_battery_is_registered_as_battery():
    device = DEVICE_TYPE_MAP["MARSTEK-BAT"](
        ip_address="192.0.2.10",
        device_type="MARSTEK-BAT",
        device_name="Marstek Venus",
        max_power_usage=800,
        energy_capacity=2080,
    )

    assert isinstance(device, MarstekBattery)
    assert isinstance(device, Battery)
    assert device.get_HWE_class().port == 50000


def test_marstek_battery_measurement_updates_scheduler_state():
    device = MarstekBattery(
        ip_address="192.0.2.10",
        device_type="MARSTEK-BAT",
        device_name="Marstek Venus",
        max_power_usage=800,
        energy_capacity=2000,
    )
    device.hwe_device = Mock(
        es_status=AsyncMock(
            return_value=MarstekESStatus(
                bat_soc=50,
                bat_cap=2080,
                pv_power=0,
                ongrid_power=300,
                offgrid_power=0,
                total_pv_energy=0,
                total_grid_output_energy=15617,
                total_grid_input_energy=32114,
                total_load_energy=0,
            )
        )
    )
    logger = Mock()

    asyncio.run(device.perform_measurement(logger))

    assert device.inst_power_usage == 300
    assert device.state_of_charge_pct == 50
    assert device.energy_capacity == 2080
    assert device.energy_stored == 1040
    assert device.policy.energy_stored_upper == 2080


def test_parse_marstek_es_status_rejects_missing_fields():
    response = json.dumps({"result": {"bat_soc": 50}}).encode()

    with pytest.raises(InvalidMarstekResponseError, match="bat_cap"):
        parse_marstek_es_status(response)


def test_parse_marstek_es_status_rejects_non_numeric_values():
    status = {
        "bat_soc": True,
        "bat_cap": 0,
        "pv_power": 0,
        "ongrid_power": 0,
        "offgrid_power": 0,
        "total_pv_energy": 0,
        "total_grid_output_energy": 0,
        "total_grid_input_energy": 0,
        "total_load_energy": 0,
    }
    response = json.dumps({"result": status}).encode()

    with pytest.raises(InvalidMarstekResponseError, match="bat_soc"):
        parse_marstek_es_status(response)


def test_marstek_battery_ignores_invalid_response_without_overwriting_state():
    device = MarstekBattery(
        ip_address="192.0.2.10",
        device_type="MARSTEK-BAT",
        device_name="Marstek Venus",
        max_power_usage=800,
        energy_capacity=2000,
    )
    device.inst_power_usage = 300
    device.state_of_charge_pct = 50
    device.hwe_device = Mock(
        es_status=AsyncMock(
            side_effect=InvalidMarstekResponseError("missing fields")
        )
    )
    logger = Mock()

    asyncio.run(device.perform_measurement(logger))

    assert device.inst_power_usage == 300
    assert device.state_of_charge_pct == 50
    logger.warning.assert_called_once()
