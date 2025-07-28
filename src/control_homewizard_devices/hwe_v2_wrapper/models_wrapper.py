# For python 3.11 (which is used for the raspberry pi), I was not able to use the newest version of the python-homewizard-energy.
# Therefore, to be able to still communicate with a Plug-In battery, I modified the existing models.py code from the github:
# https://github.com/homewizard/python-homewizard-energy/tree/main.
# Licensed under the Apache License, Version 2.0

"""Common models for HomeWizard Energy API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from awesomeversion import AwesomeVersion
from mashumaro.config import BaseConfig
from mashumaro.exceptions import MissingField
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import SerializationStrategy

from .const_wrapper import LOGGER, MODEL_TO_ID, MODEL_TO_NAME, Model
from .utils_wrapper import get_awesome_version


class AwesomeVersionSerializationStrategy(SerializationStrategy, use_annotations=True):
    """Serialization strategy for AwesomeVersion objects."""

    def serialize(self, value: AwesomeVersion) -> str:
        """Serialize AwesomeVersion object to string."""
        return str(value)  # pragma: no cover

    def deserialize(self, value: str) -> AwesomeVersion | None:
        """Deserialize string to AwesomeVersion object."""
        version = get_awesome_version(value)
        return version


class BaseModel(DataClassORJSONMixin):
    """Base model for all HomeWizard models."""

    # pylint: disable-next=too-few-public-methods
    class Config(BaseConfig):
        """Mashumaro configuration."""

        serialize_by_alias = True
        serialization_strategy = {  # noqa: RUF012
            AwesomeVersion: AwesomeVersionSerializationStrategy()
        }
        omit_none = True


class UpdateBaseModel(BaseModel):
    """Base model for all 'update' models."""

    def __post_serialize__(self, d: dict, context: dict | None = None):
        """Post serialize hook for UpdateBaseModel object."""
        _ = context  # Unused

        if not d:
            raise ValueError("No values to update")

        return d


def get_verification_hostname(model: str, serial_number: str) -> str:
    """Helper method to convert device model and serial to identifier

    The identifier is used to verify the device in the HomeWizard Energy API via HTTPS.
    """

    if model not in MODEL_TO_ID:
        raise ValueError(f"Unsupported model: {model}")

    return f"appliance/{MODEL_TO_ID[model]}/{serial_number}"


@dataclass(kw_only=True)
class Device(BaseModel):
    """Represent Device config."""

    model_name: str | None = None
    id: str | None = None

    product_name: str = field()
    product_type: str = field()
    serial: str = field()
    api_version: AwesomeVersion = field()
    firmware_version: str = field()

    @classmethod
    def __post_deserialize__(cls, obj: Device) -> Device:
        """Post deserialize hook for Device object."""
        _ = cls  # Unused

        obj.model_name = MODEL_TO_NAME.get(obj.product_type)
        obj.id = get_verification_hostname(obj.product_type, obj.serial)
        return obj


@dataclass(kw_only=True)
class Measurement(BaseModel):
    """Represent Measurement."""

    # Deprecated, use 'System'
    wifi_ssid: str | None = field(
        default=None,
    )
    wifi_strength: int | None = field(
        default=None,
    )

    # Generic
    energy_import_kwh: float | None = field(
        default=None,
    )
    energy_import_t1_kwh: float | None = field(
        default=None,
    )
    energy_import_t2_kwh: float | None = field(
        default=None,
    )
    energy_import_t3_kwh: float | None = field(
        default=None,
    )
    energy_import_t4_kwh: float | None = field(
        default=None,
    )
    energy_export_kwh: float | None = field(
        default=None,
    )
    energy_export_t1_kwh: float | None = field(
        default=None,
    )
    energy_export_t2_kwh: float | None = field(
        default=None,
    )
    energy_export_t3_kwh: float | None = field(
        default=None,
    )
    energy_export_t4_kwh: float | None = field(
        default=None,
    )

    power_w: float | None = field(
        default=None,
    )
    power_l1_w: float | None = field(
        default=None,
    )
    power_l2_w: float | None = field(
        default=None,
    )
    power_l3_w: float | None = field(
        default=None,
    )

    voltage_v: float | None = field(
        default=None,
    )
    voltage_l1_v: float | None = field(
        default=None,
    )
    voltage_l2_v: float | None = field(
        default=None,
    )
    voltage_l3_v: float | None = field(
        default=None,
    )

    current_a: float | None = field(
        default=None,
    )
    current_l1_a: float | None = field(
        default=None,
    )
    current_l2_a: float | None = field(
        default=None,
    )
    current_l3_a: float | None = field(
        default=None,
    )

    apparent_power_va: float | None = field(
        default=None,
    )
    apparent_power_l1_va: float | None = field(
        default=None,
    )
    apparent_power_l2_va: float | None = field(
        default=None,
    )
    apparent_power_l3_va: float | None = field(
        default=None,
    )

    reactive_power_var: float | None = field(
        default=None,
    )
    reactive_power_l1_var: float | None = field(
        default=None,
    )
    reactive_power_l2_var: float | None = field(
        default=None,
    )
    reactive_power_l3_var: float | None = field(
        default=None,
    )

    power_factor: float | None = field(
        default=None,
    )
    power_factor_l1: float | None = field(
        default=None,
    )
    power_factor_l2: float | None = field(
        default=None,
    )
    power_factor_l3: float | None = field(
        default=None,
    )

    frequency_hz: float | None = field(
        default=None,
    )

    # P1 Meter specific (I removed most)
    tariff: int | None = field(
        default=None,
    )

    # Battery Specific
    cycles: int | None = field(
        default=None,
    )
    state_of_charge_pct: float | None = field(
        default=None,
    )

    @staticmethod
    def to_datetime(timestamp: str | int) -> datetime:
        """Convert DSRM gas-timestamp to datetime object.

        Args:
            timestamp: Timestamp string, formatted as YYMMDDHHMMSS or YYYY-MM-DDTHH:MM:SS

        Returns:
            A datetime object.
        """
        if isinstance(timestamp, int):
            # V1 API uses int for timestamp
            return datetime.strptime(str(timestamp), "%y%m%d%H%M%S")

        return datetime.fromisoformat(timestamp)

    @staticmethod
    def hex_to_readable(value: str | None) -> str | None:
        """Convert hex string to readable string, if possible.

        Args:
            value: String to convert, e.g. '4E47475955'

        Returns:
            A string formatted or original value when failed.
        """
        try:
            return bytes.fromhex(value).decode("utf-8")
        except (TypeError, ValueError):
            return value

    @classmethod
    # pylint: disable=too-many-statements
    def __pre_deserialize__(cls, d: dict[Any, Any]) -> dict[Any, Any]:
        _ = cls  # Unused

        if "wifi_ssid" not in d:
            # This is a v2 API response, no need to remap
            return d

        d["protocol_version"] = d.get("smr_version")
        d["tariff"] = d.get("active_tariff")
        d["energy_import_kwh"] = d.get(
            "total_power_import_kwh", d.get("total_power_import_t1_kwh")
        )
        d["energy_import_t1_kwh"] = d.get("total_power_import_t1_kwh")
        d["energy_import_t2_kwh"] = d.get("total_power_import_t2_kwh")
        d["energy_import_t3_kwh"] = d.get("total_power_import_t3_kwh")
        d["energy_import_t4_kwh"] = d.get("total_power_import_t4_kwh")
        d["energy_export_kwh"] = d.get(
            "total_power_export_kwh", d.get("total_power_export_t1_kwh")
        )
        d["energy_export_t1_kwh"] = d.get("total_power_export_t1_kwh")
        d["energy_export_t2_kwh"] = d.get("total_power_export_t2_kwh")
        d["energy_export_t3_kwh"] = d.get("total_power_export_t3_kwh")
        d["energy_export_t4_kwh"] = d.get("total_power_export_t4_kwh")
        d["power_w"] = d.get("active_power_w")
        d["power_l1_w"] = d.get("active_power_l1_w")
        d["power_l2_w"] = d.get("active_power_l2_w")
        d["power_l3_w"] = d.get("active_power_l3_w")
        d["voltage_v"] = d.get("active_voltage_v")
        d["voltage_l1_v"] = d.get("active_voltage_l1_v")
        d["voltage_l2_v"] = d.get("active_voltage_l2_v")
        d["voltage_l3_v"] = d.get("active_voltage_l3_v")
        d["current_a"] = d.get("active_current_a")
        d["current_l1_a"] = d.get("active_current_l1_a")
        d["current_l2_a"] = d.get("active_current_l2_a")
        d["current_l3_a"] = d.get("active_current_l3_a")
        d["apparent_power_va"] = d.get("active_apparent_power_va")
        d["apparent_power_l1_va"] = d.get("active_apparent_power_l1_va")
        d["apparent_power_l2_va"] = d.get("active_apparent_power_l2_va")
        d["apparent_power_l3_va"] = d.get("active_apparent_power_l3_va")
        d["reactive_power_var"] = d.get("active_reactive_power_var")
        d["reactive_power_l1_var"] = d.get("active_reactive_power_l1_var")
        d["reactive_power_l2_var"] = d.get("active_reactive_power_l2_var")
        d["reactive_power_l3_var"] = d.get("active_reactive_power_l3_var")
        d["power_factor"] = d.get("active_power_factor")
        d["power_factor_l1"] = d.get("active_power_factor_l1")
        d["power_factor_l2"] = d.get("active_power_factor_l2")
        d["power_factor_l3"] = d.get("active_power_factor_l3")
        d["frequency_hz"] = d.get("active_frequency_hz")
        d["average_power_15m_w"] = d.get("active_power_average_w")
        d["monthly_power_peak_w"] = d.get("montly_power_peak_w")
        d["monthly_power_peak_timestamp"] = d.get("montly_power_peak_timestamp")
        d["external_devices"] = d.get("external_devices")

        return d

    @classmethod
    def __post_deserialize__(cls, obj: Measurement) -> Measurement:
        """Post deserialize hook for Measurement object."""
        _ = cls  # Unused

        # Some smart meters report a tariff other than 1, 2, 3 or 4, which is invalid
        if obj.tariff not in (1, 2, 3, 4):
            obj.tariff = None

        return obj


@dataclass
class SystemUpdate(UpdateBaseModel):
    """Represent System update config."""

    cloud_enabled: bool | None = field(default=None)
    status_led_brightness_pct: int | None = field(default=None)
    api_v1_enabled: bool | None = field(default=None)


@dataclass(kw_only=True)
class System(BaseModel):
    """Represent System config."""

    wifi_strength_pct: int | None = None

    wifi_ssid: str | None = field(default=None)
    wifi_rssi_db: int | None = field(default=None)
    cloud_enabled: bool | None = field(default=None)
    uptime_s: int | None = field(default=None)
    status_led_brightness_pct: int | None = field(default=None)
    api_v1_enabled: bool | None = field(default=None)

    @classmethod
    def __post_deserialize__(cls, obj: System) -> System:
        _ = cls  # Unused

        if obj.wifi_rssi_db is not None:
            obj.wifi_strength_pct = (
                0
                if obj.wifi_rssi_db <= -100 or obj.wifi_rssi_db == 0
                else 100
                if obj.wifi_rssi_db >= -50
                else 2 * (obj.wifi_rssi_db + 100)
            )

        return obj


@dataclass(kw_only=True)
class Token(BaseModel):
    """Represent Token."""

    token: str = field()
