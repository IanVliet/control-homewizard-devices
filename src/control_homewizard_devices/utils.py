import logging
from logging.handlers import TimedRotatingFileHandler
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    P1Device,
    Battery,
    AggregateBattery,
)
import json
from quartz_solar_forecast.pydantic_models import PVSite
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from datetime import datetime, date, time, timedelta
import argparse
import platform

DEVICE_TYPE_MAP = {"HWE-SKT": SocketDevice, "HWE-P1": P1Device, "HWE-BAT": Battery}


# Setting up the logger in the main function
def setup_logger():
    parser = argparse.ArgumentParser(description="Setup logger for the application.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
        "Default is INFO.",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ],
    )
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(__name__)
    file_handler = TimedRotatingFileHandler(
        "logs/app.log", when="midnight", interval=1, backupCount=7
    )
    file_handler.suffix = "%Y-%m-%d"  # add date to rotated files
    file_handler.setFormatter(formatter)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)  # Attach same formatter

    # Configure logger
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@dataclass(frozen=True)
class ZonedTime:
    hour: int
    minute: int
    tz: ZoneInfo

    def at_date(self, d: date) -> datetime:
        """
        Returns a datetime object for the given date
        with the time set to the hour and minute of this ZonedTime.
        """
        return datetime.combine(d, time(self.hour, self.minute), tzinfo=self.tz)

    def at_next_date(self, d: date) -> datetime:
        """
        Returns a datetime object for the next occurrence of this ZonedTime
        after the given date.
        If the time has already passed today, it will return the time for tomorrow.
        """
        next_date = (
            d + timedelta(days=1) if self.at_date(d) < datetime.now(self.tz) else d
        )
        return self.at_date(next_date)

    def at_previous_date(self, d: date) -> datetime:
        """
        Returns a datetime object for the previous occurrence of this ZonedTime
        before the given date.
        If the time has not yet passed today, it will return the time for yesterday.
        """
        prev_date = (
            d - timedelta(days=1) if self.at_date(d) > datetime.now(self.tz) else d
        )
        return self.at_date(prev_date)

    def get_naive_utc_date(self) -> datetime:
        """
        Returns a naive datetime object (in UTC)
        with the time based on the hour and minute of this ZonedTime.
        The date is today if the ZonedTime has already passed today,
        and yesterday if the ZonedTime has not yet passed.
        """
        today = datetime.now(self.tz).date()
        local_dt = self.at_previous_date(today)
        # Convert to UTC and make naive
        utc_dt = local_dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        return utc_dt

    def seconds_until_next(self) -> float:
        """
        Returns the number of seconds until the next occurrence of this ZonedTime.
        """
        now = datetime.now(self.tz)
        next_time = self.at_next_date(now.date())
        return (next_time - now).total_seconds()

    def __str__(self):
        return f"{self.hour:02d}:{self.minute:02d} {self.tz.key}"


def initialize_devices(config_json_filepath) -> list[CompleteDevice]:
    with open(config_json_filepath, "r") as config_file:
        config_data = json.load(config_file)

    all_devices = []
    device_data = config_data["devices"]
    for device_dict in device_data:
        device_class = DEVICE_TYPE_MAP.get(device_dict["device_type"], CompleteDevice)
        device = device_class(**device_dict)
        all_devices.append(device)

    return all_devices


def initialize_solarpanel_sites(config_json_filepath) -> dict[str, PVSite]:
    with open(config_json_filepath, "r") as config_file:
        config_data = json.load(config_file)

    solar_panels_data = config_data["solar panels"]
    sites = {}
    for solar_panel_data in solar_panels_data:
        name = solar_panel_data["name"]
        params = solar_panel_data["quartz_parameters"]
        site = PVSite(**params)
        sites[name] = site

    return sites


def is_raspberry_pi():
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        # Look for "BCM" or "Raspberry Pi" in cpuinfo
        if "BCM" in cpuinfo or "Raspberry Pi" in cpuinfo:
            return True
    except FileNotFoundError:
        return False
    return False


class ColNames:
    POWER_W = "power [w]"
    AVAILABLE_POWER = "available power"

    @staticmethod
    def state(device: CompleteDevice | AggregateBattery):
        return f"schedule {device.device_name}"

    @staticmethod
    def energy_stored(device: CompleteDevice | AggregateBattery):
        return f"energy stored {device.device_name}"


class TimelineColNames:
    PREDICTED_POWER = "predicted power [w]"
    MEASURED_POWER = "measured power [w]"

    @staticmethod
    def measured_power_consumption(device: CompleteDevice | AggregateBattery):
        return f"measured power consumption {device.device_name} [w]"

    @staticmethod
    def predicted_power_consumption(device: CompleteDevice | AggregateBattery):
        return f"predicted power consumption {device.device_name} [w]"
