import asyncio
import os
from control_homewizard_devices.utils import (
    setup_logger,
    initialize_devices,
    initialize_solarpanel_sites,
    ZonedTime,
    is_raspberry_pi,
    TimelineColNames,
)
from control_homewizard_devices.device_classes import (
    SocketDevice,
    Battery,
)
from control_homewizard_devices.schedule_devices import (
    DeviceSchedulingOptimization,
    Variables,
    ColNames,
)
from control_homewizard_devices.constants import TZ, DELTA_T, PERIODIC_SLEEP_DURATION
from contextlib import AsyncExitStack
import sys
from typing import cast

from quartz_solar_forecast.forecast import run_forecast
from datetime import datetime
import pandas as pd
import signal
import numpy as np


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
        self.curr_timeindex: datetime | None = None
        self.measurement_count: int = 0
        self.df_timeline: pd.DataFrame | None = None
        self.on_raspberry_pi = is_raspberry_pi()
        if self.on_raspberry_pi:
            from control_homewizard_devices.e_paper_display import DrawDisplay

            self.draw_display = DrawDisplay(self.socket_and_battery_list, self.logger)

    def get_forecast_data(self) -> pd.DataFrame | None:
        """
        Retrieves forecast data for the panels.
        """
        # Since the quartz library requires a naive UTC datetime,
        # we convert the ZonedTime to such a datetime object for today.
        today = self.START_FORECAST_TIME.get_naive_utc_date()
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
                "UTC"
            ).tz_convert(self.START_FORECAST_TIME.tz)
            self.logger.info("Daily power forecast retrieved successfully.")
        else:
            self.logger.error("Failed to retrieve any daily power forecast.")
        return df_solar_forecast

    def initialize_df_timeline(self):
        """
        Creates a dataframe containing the following:
        - the predicted solar power of the day
        - columns for the measured available power (history)
        - columns for each device of the set states (history)
        - columns for each device of the states to be set (prediction)
        A row entry should be added for the columns containing history.
        This should be added to every timestep of the df (e.g. 15 mins).
        The columns containing prediction will be updated periodicly.
        This only needs to happen from the current timestep onwards.
        """
        if self.df_solar_forecast is None:
            return None
        if not isinstance(self.df_solar_forecast.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected DatetimeIndex, got {type(self.df_solar_forecast.index)}"
            )
        logger = self.logger
        logger.info("Initializing df daily schedule")
        now = datetime.now(self.START_FORECAST_TIME.tz)
        next_forecast_time = self.START_FORECAST_TIME.at_next_date(now.date())
        logger.debug(f"now tz: {now.tzinfo}")
        logger.debug(f"next_forecast_time tz: {next_forecast_time.tzinfo}")
        logger.debug(
            f"df_solar_forecast index tz: {self.df_solar_forecast.index.tzinfo}"
        )
        try:
            # Slice the solar forecast data to the next forecast time.
            df_solar_slice = self.df_solar_forecast.loc[:next_forecast_time]
            new_datetime_index = pd.date_range(
                start=df_solar_slice.index.min(),
                end=df_solar_slice.index.max(),
                freq=f"{DELTA_T}h",
            )
            df_timeline = df_solar_slice.reindex(new_datetime_index).interpolate(
                method="time"
            )
            df_timeline[TimelineColNames.PREDICTED_POWER] = (
                df_timeline["power_kw"] * 1000
            )
            df_timeline[TimelineColNames.MEASURED_POWER] = np.nan
            # Initialize columns for sockets
            for device in self.sorted_sockets:
                df_timeline[TimelineColNames.measured_power_consumption(device)] = (
                    np.nan
                )
                df_timeline[TimelineColNames.predicted_power_consumption(device)] = (
                    np.nan
                )
            # Initialize columns for aggregate battery
            df_timeline[
                TimelineColNames.measured_power_consumption(
                    self.optimization.data.aggregate_battery
                )
            ] = np.nan
            df_timeline[
                TimelineColNames.predicted_power_consumption(
                    self.optimization.data.aggregate_battery
                )
            ] = np.nan
            return df_timeline
        except Exception as e:
            logger.error(
                f"Error during initialization of df_timeline with solar forecast: {e}"
            )
            return None

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
            self.df_timeline = None
            for socket in self.sorted_sockets:
                socket.energy_stored = 0.0
            self.logger.info(
                "A new day has arrived so the energy stored "
                "in the sockets is set back to 0.0"
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
            self.update_df_timeline(-total_power)
            if self.on_raspberry_pi:
                logger.info("Update E-paper display")
                # Update display
                self.draw_display.draw_full_update(
                    self.df_timeline, self.curr_timeindex
                )

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
                    "Predicted available solar power "
                    f"{df_solar_slice.iloc[0].item() * 1000} W"
                )
                logger.info(
                    "Solar forecast data available "
                    f"for the next {len(df_solar_slice)} time steps."
                )
                # Available power is defined as - total power.
                diff = df_solar_slice.iloc[0] + total_power / 1000  # Convert kW to W
                logger.info(f"Power difference: {diff.item() * 1000} W")
                df_power_prediction = df_solar_slice - diff
                logger.debug(f"Power prediction DataFrame:\n{df_power_prediction}")

                data, results = self.optimization.solve_schedule_devices(
                    df_power_prediction, self.socket_and_battery_list, overcharge=True
                )
                self.df_power_interpolated = data.df_power_interpolated
                self.df_schedule = results[-1].df_variables
                self.schedule_determine_socket_states(results)
                logger.info(
                    "===== socket states determined based on schedule "
                    "produced with solar forecast ====="
                )
                return
            except Exception as e:
                logger.error(f"Error during scheduling with solar forecast: {e}")

            if self.df_solar_forecast is None:
                logger.warning(
                    "No solar forecast data available, "
                    "using simple scheduling based on currently available power."
                )
            elif len(self.df_solar_forecast) == 0:
                logger.warning(
                    "Solar forecast data is empty, "
                    "using simple scheduling based on currently available power."
                )
            else:
                logger.info(
                    "Falling back to simple scheduling "
                    "based on currently available power."
                )
            self.simple_determine_socket_states(total_power)
            logger.info(
                "===== socket states determined "
                "based on currently available power ====="
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

    def update_df_timeline(self, available_power: float):
        logger = self.logger
        logger.info("Updating data stored in df_timeline")
        if self.df_timeline is None:
            logger.info("Attempting to initialize df_timeline")
            self.df_timeline = self.initialize_df_timeline()
            if self.df_timeline is None:
                logger.warning(
                    "Updating failed, since df_timeline has not initialized properly. "
                )
                return

        timeindex, pos_curr_timestep = self.get_curr_timeindex()
        if timeindex is None:
            # logging message is present in get_curr_timeindex
            return
        total_inst_power_usage_aggregate_battery = (
            self.get_total_inst_power_usage_aggregate_battery()
        )
        self.update_moving_average_columns(
            timeindex, available_power, total_inst_power_usage_aggregate_battery
        )

        next_timeindex = self.get_next_timeindex(pos_curr_timestep)
        if next_timeindex is None:
            # Logging message is present in get_next_timeindex
            return
        # Update the predicted values
        for device in self.optimization.data.devices_list:
            logger.debug(
                f"Updating predicted power consumption for {device.device_name} "
                "in df_timeline from states in df_schedule"
            )
            self.df_timeline.loc[
                next_timeindex:, TimelineColNames.predicted_power_consumption(device)
            ] = (
                device.max_power_usage
                * self.df_schedule.loc[next_timeindex:, ColNames.state(device)]
            )

    def get_curr_timeindex(self) -> tuple[datetime | None, int]:
        if self.df_timeline is None:
            error_msg = "Cannot find timeindex while df_timeline is None"
            self.logger.error(error_msg)
            raise IndexError(error_msg)
        now = datetime.now(self.START_FORECAST_TIME.tz)
        pos_curr_timestep = self.df_timeline.index.searchsorted(now, side="right")
        if isinstance(pos_curr_timestep, list):
            self.logger.warning(
                "The position of the index of the current timestep is not as excepted."
                "Instead of an int it was a list of int."
                "Using the first value for the position"
            )
            pos_curr_timestep = pos_curr_timestep[0]
        pos_curr_timestep = int(pos_curr_timestep)
        if pos_curr_timestep >= len(self.df_timeline.index):
            self.logger.warning(
                "A timestamp in df_timeline for the current time could not be found."
                "A new dataframe should be initialized soon."
            )
            return None, pos_curr_timestep
        timeindex = self.df_timeline.index[pos_curr_timestep]
        return timeindex, pos_curr_timestep

    def get_next_timeindex(self, pos_curr_timestep: int):
        if self.df_timeline is None:
            error_msg = "Cannot find next timeindex while df_timeline is None"
            self.logger.error(error_msg)
            raise IndexError(error_msg)
        next_pos_timestep = pos_curr_timestep + 1
        if next_pos_timestep >= len(self.df_timeline.index):
            self.logger.info(
                "There is no next timeindex in df_timeline. "
                "Therefore, the next df_timeline should be initialized soon."
            )
            return None
        next_timeindex = self.df_timeline.index[next_pos_timestep]
        return next_timeindex

    def get_total_inst_power_usage_aggregate_battery(self):
        length_battery_list = len(self.optimization.data.battery_list)
        if length_battery_list == 0:
            return 0.0
        total_inst_power_usage_aggregate_battery = 0.0
        for battery in self.optimization.data.battery_list:
            if battery.inst_power_usage is None:
                self.logger.warning(
                    f"{battery.device_name}'s inst_power_usage is None. "
                    "Skipping for the calculation of the total inst_power_usage "
                    "of the aggregate battery"
                )
                continue
            total_inst_power_usage_aggregate_battery += battery.inst_power_usage
        return total_inst_power_usage_aggregate_battery / length_battery_list

    def update_moving_average_columns(
        self,
        timeindex: datetime,
        available_power: float,
        total_inst_power_usage_aggregate_battery: float,
    ):
        logger = self.logger
        if self.df_timeline is None:
            error_msg = (
                "update_moving_average_columns should only be called "
                "after it has been made that df_timeline is not None"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.curr_timeindex is None or self.curr_timeindex != timeindex:
            logger.debug(f"Initializizing moving averages for {timeindex}")
            self.curr_timeindex = timeindex
            self.measurement_count = 1
            # New timeindex: start the moving average with the first value
            self.df_timeline.at[timeindex, TimelineColNames.MEASURED_POWER] = (
                available_power
            )
            for device in self.sorted_sockets:
                if device.inst_power_usage is None:
                    logger.warning(
                        "Initialization of moving average failed for socket "
                        f"{device.device_name}, "
                        "since inst_power_usage is None. Skipping..."
                    )
                    continue
                self.df_timeline.at[
                    timeindex, TimelineColNames.measured_power_consumption(device)
                ] = device.inst_power_usage

            self.df_timeline.at[
                timeindex,
                TimelineColNames.measured_power_consumption(
                    self.optimization.data.aggregate_battery
                ),
            ] = total_inst_power_usage_aggregate_battery
        else:
            # Must still be the same timeindex --> Updating moving average
            logger.debug(
                f"Updating moving averages of {timeindex} with the measured powers"
            )
            self.measurement_count += 1
            self.update_moving_average(
                timeindex, TimelineColNames.MEASURED_POWER, available_power
            )
            for device in self.optimization.data.devices_list:
                if isinstance(device, SocketDevice):
                    if device.inst_power_usage is None:
                        self.logger.warning(
                            "Updating of moving average failed for socket "
                            f"{device.device_name}, "
                            "since inst_power_usage is None. Skipping..."
                        )
                        continue
                    self.update_moving_average(
                        timeindex,
                        TimelineColNames.measured_power_consumption(device),
                        device.inst_power_usage,
                    )
                else:
                    self.update_moving_average(
                        timeindex,
                        TimelineColNames.measured_power_consumption(device),
                        total_inst_power_usage_aggregate_battery,
                    )

    def update_moving_average(
        self, timeindex: datetime, col_name: str, new_value: float
    ):
        self.logger.debug(f"Updating moving average for {col_name}")
        if self.df_timeline is None:
            self.logger.warning("df_timeline has not yet been initialized")
            return
        old_avg: float = cast(
            float, self.df_timeline.at[pd.Timestamp(timeindex), col_name]
        )
        self.df_timeline.at[timeindex, col_name] = (
            old_avg + (new_value - old_avg) / self.measurement_count
        )

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
        if controller.on_raspberry_pi:
            controller.draw_display.clear_sleep_display()


async def main(controller: DeviceController):
    async with AsyncExitStack() as stack:
        hwe_devices = []
        for device in controller.all_devices:
            device.hwe_device = await stack.enter_async_context(device.get_HWE_class())
            hwe_devices.append(device.hwe_device)

        loop = asyncio.get_running_loop()
        main_task = asyncio.create_task(main_loop(controller))

        def shutdown(sig: signal.Signals):
            controller.logger.info(f"Received {sig.name}, cancelling main loop...")
            main_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown, sig)
                controller.logger.info(
                    f"Succesfully added signal hanndler for {sig.name}."
                )
            except NotImplementedError:
                controller.logger.info(
                    f"On windows, skipped adding signal handler for {sig.name}"
                )
                pass

        try:
            await main_task
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
