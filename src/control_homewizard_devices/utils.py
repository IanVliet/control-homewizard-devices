import logging
from logging.handlers import TimedRotatingFileHandler
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    P1Device,
    Battery,
)
import json
from quartz_solar_forecast.pydantic_models import PVSite
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from datetime import datetime, date, time, timedelta
import argparse

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

    def at_naive_date(self, d: date) -> datetime:
        """
        Returns a naive datetime object for the given date
        with the time set to the hour and minute of this ZonedTime.
        """
        return datetime.combine(d, time(self.hour, self.minute))

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
