import asyncio
from unittest.mock import AsyncMock, Mock

from control_homewizard_devices.device_classes import Battery, MarstekBattery
from control_homewizard_devices.marstek_api.udp_request import MarstekESStatus
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
