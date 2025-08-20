import asyncio
import os
from .utils import (
    setup_logger,
    initialize_devices,
    initialize_solarpanel_sites,
    ZonedTime,
)
from .device_classes import (
    SocketDevice,
    Battery,
)
from .schedule_devices import (
    DeviceSchedulingOptimization,
    Variables,
    ColNames,
)
from .constants import TZ, DELTA_T, PERIODIC_SLEEP_DURATION
from contextlib import AsyncExitStack
import sys

from quartz_solar_forecast.forecast import run_forecast
from datetime import datetime
import pandas as pd


CONFIG_ENV_VAR = "MYPROJECT_CONFIG_PATH"

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "config_devices.json"
)


def get_config_path() -> str:
    """Returns the absolute path to the config file, using an env var or default."""
    return os.path.abspath(os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


class DeviceController:
    def __init__(self):
        self.START_FORECAST_TIME = ZonedTime(6, 0, TZ)
        self.config_path = get_config_path()
        self.all_devices = initialize_devices(self.config_path)
        self.solar_panel_sites = initialize_solarpanel_sites(self.config_path)
        self.logger = setup_logger()
        self.socket_and_battery_list = [
            device
            for device in self.all_devices
            if isinstance(device, SocketDevice) or isinstance(device, Battery)
        ]
        self.sorted_sockets = sorted(
            [
                device
                for device in self.socket_and_battery_list
                if isinstance(device, SocketDevice)
            ],
            key=lambda d: d.priority,
        )
        self.df_solar_forecast: pd.DataFrame | None = self.get_forecast_data()
        self.optimization = DeviceSchedulingOptimization(DELTA_T)

    def get_forecast_data(self) -> pd.DataFrame | None:
        """
        Retrieves forecast data for the panels.
        """
        today = self.START_FORECAST_TIME.at_naive_date(datetime.now().date())
        df_solar_forecast = None
        solar_panel_sites = self.solar_panel_sites
        logger = self.logger
        for name, site in solar_panel_sites.items():
            try:
                df_forecast = run_forecast(site=site, ts=today, nwp_source="icon")

                logger.info(f"Retrieved forecast data for {name}")
                if df_solar_forecast is None:
                    df_solar_forecast = df_forecast.copy()
                else:
                    df_solar_forecast = df_solar_forecast.add(df_forecast, fill_value=0)
            except Exception as e:
                logger.error(f"Error retrieving forecast data for {name}: {e}")
                continue
        if df_solar_forecast is not None:
            if not isinstance(df_solar_forecast.index, pd.DatetimeIndex):
                logger.error(
                    f"Forecast index for {name} is not a DatetimeIndex, raising..."
                )
                raise TypeError(
                    f"Expected DatetimeIndex, got {type(df_forecast.index)}"
                )
            df_solar_forecast.index = df_solar_forecast.index.tz_localize(
                self.START_FORECAST_TIME.tz
            )
            self.logger.info("Daily power forecast retrieved successfully.")
        else:
            self.logger.error("Failed to retrieve any daily power forecast.")
        return df_solar_forecast

    async def daily_power_forecast(self):
        """
        Retrieves the power forecast for all solar panels on a daily basis.
        """
        while True:
            delay = self.START_FORECAST_TIME.seconds_until_next()
            self.logger.info(
                f"Waiting for {delay} seconds until next forecast update..."
            )
            await asyncio.sleep(delay)

            self.df_solar_forecast = self.get_forecast_data()
            for socket in self.sorted_sockets:
                socket.energy_stored = 0.0
            self.logger.info(
                "A new day has arrived so the energy stored in the sockets is set back to 0.0"
            )

    async def periodic_schedule_update(self):
        """
        Periodically updates the schedule for all devices.
        """
        logger = self.logger
        while True:
            async with asyncio.TaskGroup() as tg:
                for device in self.all_devices:
                    tg.create_task(device.perform_measurement(logger))
            logger.info("===== devices info gathered and updated =====")
            total_power = self.get_total_available_power()
            logger.info(f"Total available power: {-total_power} W")

            self.primary_scheduling_with_fallback(total_power)

            async with asyncio.TaskGroup() as tg:
                for socket in self.sorted_sockets:
                    tg.create_task(socket.update_power_state(logger))
            logger.info("===== socket states updated =====")

            await asyncio.sleep(PERIODIC_SLEEP_DURATION)

    def get_total_available_power(self) -> float:
        return sum(
            power
            for power in (
                device.get_instantaneous_power() for device in self.all_devices
            )
            if power is not None
        )

    def primary_scheduling_with_fallback(self, total_power: float):
        logger = self.logger
        now = datetime.now(self.START_FORECAST_TIME.tz)
        next_forecast_time = self.START_FORECAST_TIME.at_next_date(now.date())
        if self.df_solar_forecast is not None:
            logger.debug(f"now tz: {now.tzinfo}")
            logger.debug(f"next_forecast_time tz: {next_forecast_time.tzinfo}")
            if not isinstance(self.df_solar_forecast.index, pd.DatetimeIndex):
                raise TypeError(
                    f"Expected DatetimeIndex, got {type(self.df_solar_forecast.index)}"
                )
            logger.debug(
                f"df_solar_forecast index tz: {self.df_solar_forecast.index.tzinfo}"
            )
            try:
                # Slice the solar forecast data to the next forecast time.
                df_solar_slice = self.df_solar_forecast.loc[now:next_forecast_time]
                logger.info(
                    f"Predicted available solar power {df_solar_slice.iloc[0].item() * 1000} W"
                )
                logger.info(
                    f"Solar forecast data available for the next {len(df_solar_slice)} time steps."
                )
                # Available power is defined as - total power.
                diff = df_solar_slice.iloc[0] + total_power / 1000  # Convert kW to W
                logger.info(f"Power difference: {diff.item() * 1000} W")
                df_power_prediction = df_solar_slice - diff
                logger.debug(f"Power prediction DataFrame:\n{df_power_prediction}")

                data, results = self.optimization.solve_schedule_devices(
                    df_power_prediction, self.socket_and_battery_list
                )
                self.schedule_determine_socket_states(results)
                logger.info(
                    "===== socket states determined based on schedule produced with solar forecast ====="
                )
                return
            except Exception as e:
                logger.error(f"Error during scheduling with solar forecast: {e}")

            if self.df_solar_forecast is None:
                logger.warning(
                    "No solar forecast data available, using simple scheduling based on currently available power."
                )
            elif len(self.df_solar_forecast) == 0:
                logger.warning(
                    "Solar forecast data is empty, using simple scheduling based on currently available power."
                )
            else:
                logger.info(
                    "Falling back to simple scheduling based on currently available power."
                )
            self.simple_determine_socket_states(total_power)
            logger.info(
                "===== socket states determined based on currently available power ====="
            )

    def schedule_determine_socket_states(self, results: list[Variables]):
        main_result = results[-1]
        df_schedule = main_result.df_variables
        current_schedule = df_schedule.iloc[0]
        for socket in self.sorted_sockets:
            if current_schedule[ColNames.state(socket)] > 0:
                socket.updated_state = True
            else:
                socket.updated_state = False

    def simple_determine_socket_states(self, total_power):
        # available power is defined as - total power
        available_power = -total_power
        sorted_sockets = self.sorted_sockets

        for socket in sorted_sockets:
            if socket.should_power_on(available_power):
                socket.updated_state = True
                available_power -= socket.max_power_usage
            else:
                socket.updated_state = False


async def main_loop(
    controller: DeviceController,
):
    logger = controller.logger
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(controller.daily_power_forecast())
            tg.create_task(controller.periodic_schedule_update())

    except asyncio.CancelledError:
        logger.info("Loop cancelled gracefully...")
        raise
    finally:
        logger.info("Cleaning up before shutdown...")


async def main(controller: DeviceController):
    async with AsyncExitStack() as stack:
        hwe_devices = []
        for device in controller.all_devices:
            device.hwe_device = await stack.enter_async_context(device.get_HWE_class())
            hwe_devices.append(device.hwe_device)

        try:
            await main_loop(controller)
        except asyncio.CancelledError:
            controller.logger.info("Main loop cancelled, shutting down...")
        except Exception as e:
            controller.logger.error(f"An error occurred in main loop: {e}")
            # Optionally, you can re-raise the exception if you want it to propagate
            raise
        finally:
            controller.logger.info("Shutting down...")

        controller.logger.info("Shutdown complete")
        sys.exit(130)


def entrypoint():
    controller = DeviceController()
    asyncio.run(main(controller))


if __name__ == "__main__":
    entrypoint()
