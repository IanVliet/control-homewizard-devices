import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from control_homewizard_devices.device_classes import SocketDevice, Battery
from control_homewizard_devices.utils import is_raspberry_pi, TimelineColNames
from control_homewizard_devices.constants import (
    ICON_SIZE,
    FONT_SIZE_LARGE,
    FONT_SIZE_SMALL,
)
import pandas as pd
import numpy as np
from datetime import datetime

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
LIBS_PATH = os.path.join(REPO_ROOT, "libs")
waveshare_lib = os.path.join(
    LIBS_PATH, "e-Paper", "RaspberryPi_JetsonNano", "python", "lib"
)
icons_dir = os.path.join(REPO_ROOT, "icons")

if os.path.exists(waveshare_lib) and waveshare_lib not in sys.path:
    sys.path.append(waveshare_lib)

FONTS_DIR = os.path.join(REPO_ROOT, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)
font_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")

if is_raspberry_pi():
    from waveshare_epd import epd4in2_V2  # noqa: E402

    class DrawDisplay:
        def __init__(
            self, devices: list[SocketDevice | Battery], logger: logging.Logger
        ) -> None:
            self.devices = devices
            self.logger = logger
            self.resized_icons = self.get_resized_icons()
            self.all_resized_icons = {
                (device, ICON_SIZE): icon for device, icon in self.resized_icons.items()
            }
            try:
                logger.info("Setting up E-paper display class")
                self.epd = epd4in2_V2.EPD()
                logger.info("Init and clear E-paper display")
                self.epd.init()
                self.epd.Clear()
                self.font_large = ImageFont.truetype(font_path, size=FONT_SIZE_LARGE)
                self.font_small = ImageFont.truetype(font_path, size=FONT_SIZE_SMALL)
                (
                    self.positions,
                    cols,
                    rows,
                    self.cell_width,
                    self.cell_height,
                    self.max_text_width,
                    self.height_all_icons,
                ) = self.grid_positions()
            except IOError as e:
                logger.error(f"Setup E-paper display failed with error: {e}")
            except asyncio.CancelledError:
                logger.info("Setup E-paper display cancelled")
                epd4in2_V2.epdconfig.module_exit(cleanup=True)
                raise

        def get_resized_icons(self):
            self.logger.info("Loading icon files")
            resized_icons: dict[SocketDevice | Battery, Image.Image] = {}
            for device in self.devices:
                icon_path = os.path.join(
                    icons_dir, f"{device.device_name.replace(' ', '_')}.bmp"
                )
                icon = Image.open(icon_path)
                resized = icon.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
                resized_icons[device] = resized
            return resized_icons

        def grid_positions(self):
            self.logger.info(
                "Calculating positions for cells containing charge percentages "
                "and icons"
            )
            max_text_width, text_height = get_text_width_and_height(
                "100%", self.font_large
            )
            # calculate grid for icon + text
            cell_width = max_text_width + ICON_SIZE
            cell_height = max(ICON_SIZE, text_height)
            n_cells = len(self.devices)

            max_cols = self.epd.width // cell_width
            cols = min(max_cols, n_cells)

            total_icons_width = cols * cell_width
            if cols > 1:
                spacing = (self.epd.width - total_icons_width) // (cols - 1)
            else:
                spacing = 0

            rows = math.ceil(n_cells / cols)
            total_height = rows * cell_height

            positions: dict[SocketDevice | Battery, tuple[int, int]] = {}

            for idx, device in enumerate(self.devices):
                row = idx // cols
                col = idx % cols
                x = col * (cell_width + spacing)
                y = row * cell_height
                positions[device] = (x, y)

            return (
                positions,
                cols,
                rows,
                cell_width,
                cell_height,
                max_text_width,
                total_height,
            )

        def create_image_full_plot(
            self, df_timeline: pd.DataFrame, curr_timeindex: datetime
        ) -> Image.Image:
            if not isinstance(df_timeline.index, pd.DatetimeIndex):
                error_msg = (
                    "Expected DatetimeIndex when drawing plot, "
                    f"got {type(df_timeline.index)}"
                )
                self.logger.error(error_msg)
                raise TypeError(error_msg)
            epd = self.epd
            canvas_width, canvas_height = (
                epd.width,
                epd.height - self.height_all_icons,
            )
            plot_image = Image.new("L", (canvas_width, canvas_height), epd.GRAY1)
            plot_draw = ImageDraw.Draw(plot_image)

            # Calculate max height for small font
            ascent, descent = self.font_small.getmetrics()
            max_height_small_font = ascent + descent
            # Draw label power for y-axis
            y_label_text = "power [kW]"
            y_label_w, y_label_h = get_text_width_and_height(
                y_label_text, self.font_small
            )
            y_label_image = Image.new(
                "L", (y_label_w, max_height_small_font), epd.GRAY1
            )
            y_label_draw = ImageDraw.Draw(y_label_image)
            y_label_draw.text(
                (0, 0), y_label_text, fill=epd.GRAY4, font=self.font_small
            )
            # Rotate the y-label and add it to the plot
            rotated_y_label = y_label_image.rotate(90, expand=True)
            x_pos_y_label = 0
            y_pos_y_label = math.ceil((canvas_height - y_label_w) / 2)
            plot_image.paste(rotated_y_label, (x_pos_y_label, y_pos_y_label))
            # Calculate position x position and space needed for x-axis label
            x_label_text = "Time"
            x_label_w, x_label_h = get_text_width_and_height(
                x_label_text, self.font_small
            )
            # Determine the ticks and labels for the x-axis
            hour_format = "%H:%M"
            start_time_tick = df_timeline.index[0]
            end_time_tick = df_timeline.index[-1]
            formatted_start_time = start_time_tick.strftime(hour_format)
            start_time_w, start_time_h = get_text_width_and_height(
                formatted_start_time, self.font_small
            )
            formatted_end_time = end_time_tick.strftime(hour_format)
            end_time_w, end_time_h = get_text_width_and_height(
                formatted_end_time, self.font_small
            )

            self.logger.debug("Formating current time label")
            formatted_current_time = curr_timeindex.strftime(hour_format)
            curr_time_label_w, curr_time_label_h = get_text_width_and_height(
                formatted_current_time, self.font_small
            )
            # Calculate max height needed for different x-axis ticks and labels
            h_padding = 1
            min_power, max_power = self.calculate_min_max_power(df_timeline)
            self.logger.debug("Min power: %s, Max power: %s", min_power, max_power)
            (
                lower_tick_y,
                upper_tick_y,
                formatted_lower_tick_y,
                formatted_upper_tick_y,
            ) = self.get_y_axis_tick_values(min_power, max_power)

            self.logger.debug(
                "Y-axis ticks: lower: %s, upper: %s",
                formatted_lower_tick_y,
                formatted_upper_tick_y,
            )
            formatted_lower_tick_y_w, formatted_lower_tick_y_h = (
                get_text_width_and_height(formatted_lower_tick_y, self.font_small)
            )
            formatted_upper_tick_y_w, formatted_upper_tick_y_h = (
                get_text_width_and_height(formatted_upper_tick_y, self.font_small)
            )

            # Calculate max width needed for y-axis
            max_y_label_w = max(
                max_height_small_font,
                math.ceil(start_time_w / 2) + h_padding,
                formatted_lower_tick_y_w + 2 * h_padding,
                formatted_upper_tick_y_w + 2 * h_padding,
            )
            # --- Draw data ---
            top_left_point = (max_y_label_w, 0)
            # Calculate the area for the plot
            plot_width, plot_height = (
                canvas_width - max_y_label_w,
                canvas_height - max_height_small_font,
            )
            # Convert the time series to pixel positions
            num_datapoints = len(df_timeline.index)
            x_pixels = np.linspace(0, plot_width - 1, num_datapoints)
            # Initialize the data image with white background
            data_image = Image.new("L", (plot_width, plot_height), color=epd.GRAY1)
            data_draw = ImageDraw.Draw(data_image)

            # Draw line for predicted power
            self.logger.debug("Drawing predicted power line")
            predicted_power = df_timeline[TimelineColNames.PREDICTED_POWER].to_numpy()
            predicted_power_points = power_array_to_points(
                predicted_power, min_power, max_power, plot_height, x_pixels
            )
            draw_line_segments(
                data_draw, predicted_power_points, fill=epd.GRAY2, width=2
            )

            self.logger.debug("Current timeindex: %s", curr_timeindex)

            # Get the position of the current time index
            if curr_timeindex in df_timeline.index:
                curr_time_pos = int(df_timeline.index.get_indexer([curr_timeindex])[0])
            else:
                curr_time_pos = None

            if curr_time_pos is not None and 0 <= curr_time_pos < num_datapoints:
                self.logger.debug(
                    "Drawing current time line at index: %s", curr_time_pos
                )
                curr_time_x = x_pixels[curr_time_pos]
                data_draw.line(
                    [(curr_time_x, 0), (curr_time_x, plot_height - 1)],
                    fill=epd.GRAY3,
                )
                self.logger.debug("Drawing current time label")
                x_pos_current_time = (
                    curr_time_x - curr_time_label_w // 2 + max_y_label_w
                )
                plot_draw.text(
                    (
                        x_pos_current_time,
                        plot_height,
                    ),
                    formatted_current_time,
                    fill=epd.GRAY4,
                    font=self.font_small,
                )

            if curr_time_pos is not None:
                measured_power = df_timeline[TimelineColNames.MEASURED_POWER].to_numpy()
                # Should go from the first not-nan value to the current time position
                if measured_power.size == 0 or np.isnan(measured_power).all():
                    self.logger.debug(
                        "All measured power values are NaN "
                        "or the array is empty "
                        "skipping drawing measured power"
                    )
                else:
                    measured_power_points = power_array_to_points(
                        measured_power[: curr_time_pos + 1],
                        min_power,
                        max_power,
                        plot_height,
                        x_pixels[: curr_time_pos + 1],
                    )

                    self.logger.debug("Drawing scheduled devices")
                    self.draw_device_schedule(
                        data_draw,
                        min_power,
                        max_power,
                        plot_height,
                        x_pixels,
                        df_timeline,
                        curr_time_pos,
                    )
                    self.logger.debug(
                        "Drawing measured power up to index: %s",
                        curr_time_pos,
                    )
                    draw_line_segments(
                        data_draw, measured_power_points, fill=epd.GRAY4, width=2
                    )

            # Draw a vertical line for the current time

            plot_image.paste(data_image, top_left_point)
            # --- Draw lines for axes ---
            # Calculate the points for the zero line
            zero_height = self.power_value_to_y_pixel(
                0, min_power, max_power, plot_height
            )
            zero_left_point = (max_y_label_w, zero_height)
            zero_right_point = (canvas_width, zero_height)
            # Draw the line where the power is 0 (y=0)
            plot_draw.line((zero_left_point, zero_right_point), fill=epd.GRAY4)

            # Draw the y-axis line
            plot_draw.line(
                (top_left_point, (max_y_label_w, plot_height - 1)),
                fill=epd.GRAY4,
            )
            # Draw x-line at the bottom
            plot_draw.line(
                ((max_y_label_w, plot_height - 1), (canvas_width, plot_height - 1)),
                fill=epd.GRAY4,
            )
            # Draw label time for x-axis
            x_pos_x_label = math.ceil((canvas_width - x_label_w) / 2)
            try:
                x_pos_x_label = self.get_position_label_avoid_overlap(
                    x_pos_current_time,
                    curr_time_label_w,
                    x_label_w,
                    canvas_width,
                    h_padding,
                )
            except NameError:
                self.logger.error(
                    "No position for current time tick, "
                    "so no overlap between tick for current time and x-label possible"
                )
            plot_draw.text(
                (x_pos_x_label, plot_height),
                x_label_text,
                fill=epd.GRAY4,
                font=self.font_small,
            )
            # Draw the start and end tick on the x-axis.
            # Only in case they do not overlap with the current tick.
            start_time_x = x_pixels[0] + max_y_label_w
            x_pos_start_time = start_time_x - start_time_w // 2
            max_x_pos_start_time = max(0, x_pos_start_time)
            if x_pos_current_time > max_x_pos_start_time + start_time_w:
                self.logger.debug("Drawing start time tick")
                plot_draw.line(
                    [(start_time_x, plot_height - 1), (start_time_x, plot_height + 1)],
                    fill=epd.GRAY4,
                )
                plot_draw.text(
                    (max_x_pos_start_time, plot_height),
                    formatted_start_time,
                    fill=epd.GRAY4,
                    font=self.font_small,
                )
            end_time_x = x_pixels[-1] + max_y_label_w
            x_pos_end_time = end_time_x - end_time_w // 2
            min_x_pos_end_tick = min(canvas_width - end_time_w, x_pos_end_time)
            if x_pos_current_time < min_x_pos_end_tick:
                # Draw a small line at the exact position
                # Draw the text as close as possible without falling off the display
                self.logger.debug("Drawing end time tick")
                plot_draw.line(
                    [(end_time_x, plot_height - 1), (end_time_x, plot_height + 1)],
                    fill=epd.GRAY4,
                )
                plot_draw.text(
                    (min_x_pos_end_tick, plot_height),
                    formatted_end_time,
                    fill=epd.GRAY4,
                    font=self.font_small,
                )

            # Draw the max tick on the y-axis
            y_pos_upper = self.power_value_to_y_pixel(
                upper_tick_y, min_power, max_power, plot_height
            )
            max_y_pos_y_upper_tick = max(
                y_pos_upper - formatted_upper_tick_y_h // 2,
                0,
            )
            plot_draw.line(
                [(max_y_label_w - 2, y_pos_upper), (max_y_label_w, y_pos_upper)],
                fill=epd.GRAY4,
            )
            plot_draw.text(
                (
                    0,
                    max_y_pos_y_upper_tick,
                ),
                formatted_upper_tick_y,
                fill=epd.GRAY4,
                font=self.font_small,
            )
            # Draw the min tick on the y-axis
            y_pos_lower = self.power_value_to_y_pixel(
                lower_tick_y, min_power, max_power, plot_height
            )
            min_y_pos_y_lower_tick = min(
                y_pos_lower - formatted_lower_tick_y_h // 2,
                plot_height - formatted_lower_tick_y_h,
            )
            plot_draw.line(
                [(max_y_label_w - 2, y_pos_lower), (max_y_label_w, y_pos_lower)],
                fill=epd.GRAY4,
            )
            plot_draw.text(
                (
                    0,
                    min_y_pos_y_lower_tick,
                ),
                formatted_lower_tick_y,
                fill=epd.GRAY4,
                font=self.font_small,
            )
            # Draw tick for 0 line.

            return plot_image

        def calculate_min_max_power(self, df_timeline: pd.DataFrame):
            predicted_power = df_timeline[TimelineColNames.PREDICTED_POWER].to_numpy()
            measured_power = df_timeline[TimelineColNames.MEASURED_POWER].to_numpy()
            sum_predicted_devices = (
                df_timeline[
                    [
                        TimelineColNames.predicted_power_consumption(device)
                        for device in self.devices
                        if isinstance(device, SocketDevice)
                    ]
                ]
                .sum(axis=1)
                .to_numpy()
            )
            sum_measured_devices = (
                df_timeline[
                    [
                        TimelineColNames.measured_power_consumption(device)
                        for device in self.devices
                        if isinstance(device, SocketDevice)
                    ]
                ]
                .sum(axis=1)
                .to_numpy()
            )
            # Get the maximum of the predicted and measured power and the devices
            max_measured = np.nanmax(measured_power, initial=0)
            max_predicted = np.nanmax(predicted_power, initial=0)
            self.logger.debug(
                "Max measured power: %s, Max predicted power: %s",
                max_measured,
                max_predicted,
            )
            max_measured_devices = np.nanmax(sum_measured_devices, initial=0)
            max_predicted_devices = np.nanmax(sum_predicted_devices, initial=0)
            self.logger.debug(
                "Max sum of power of measured devices: %s, "
                "Max sum of power of predicted devices: %s",
                max_measured_devices,
                max_predicted_devices,
            )
            max_stacked_power = np.array(
                [
                    max_predicted,
                    max_measured,
                    max_measured_devices,
                    max_predicted_devices,
                    0,
                ]
            )  # Ensure max power is atleast 0
            max_power = np.nanmax(max_stacked_power)

            min_measured = np.nanmin(measured_power, initial=0)
            min_predicted = np.nanmin(predicted_power, initial=0)
            self.logger.debug(
                "Min measured power: %s, Min predicted power: %s",
                min_measured,
                min_predicted,
            )
            min_measured_devices = np.nanmax(sum_measured_devices, initial=0)
            min_predicted_devices = np.nanmin(sum_predicted_devices, initial=0)
            self.logger.debug(
                "Min sum of power of measured devices: %s, "
                "Min sum of power of predicted devices: %s",
                min_measured_devices,
                min_predicted_devices,
            )
            min_stacked_power = np.array(
                [
                    min_predicted,
                    min_measured,
                    min_measured_devices,
                    min_predicted_devices,
                    0,
                ]
            )  # Ensure min power is atleast 0
            min_power = np.nanmin(min_stacked_power)
            return min_power, max_power

        def power_value_to_y_pixel(
            self,
            power_value: float,
            min_power: float,
            max_power: float,
            plot_height: int,
        ):
            normalized_power = (power_value - min_power) / (max_power - min_power)
            y_pixel = (1 - normalized_power) * (plot_height - 1)
            return y_pixel

        def get_position_label_avoid_overlap(
            self,
            tick_start_pos: int,
            tick_length: int,
            label_length: int,
            total_length: int,
            padding: int,
        ):
            """
            Calculates position of label to avoid other text (tick).
            Also accounts for padding around the tick text.
            Two cases for overlap:
            1. Tick is left of center and would overlap with label
            (End of tick is past the start of the label)
            --> move label to the right
            2. Tick is right of center and would overlap with label
            (End of label is past the start of the tick)
            --> move label to the left
            """
            init_label_start_pos = (total_length - label_length) // 2
            init_label_end_pos = init_label_start_pos + label_length
            tick_end_pos = tick_start_pos + tick_length
            # Add spacing between label and tick
            diff_tick_end_to_label = init_label_start_pos - (tick_end_pos + padding)
            diff_label_end_to_label = (tick_start_pos - padding) - init_label_end_pos
            overlap = diff_tick_end_to_label < 0 and diff_label_end_to_label < 0
            left_sided = diff_tick_end_to_label >= diff_label_end_to_label
            if overlap and left_sided:
                pos_label = init_label_start_pos - diff_tick_end_to_label
            elif overlap and not left_sided:
                pos_label = init_label_start_pos + diff_label_end_to_label
            else:
                pos_label = init_label_start_pos
            return pos_label

        def get_y_axis_tick_values(self, min_power: float, max_power: float):
            """
            Get an upper and lower tick value for the y-axis.
            First an attempt is made to get a multiple of 1000.
            If that is not possible a multiple of 100 is used.
            """
            # Has to round down for upper tick (so it doesn't fall off the image)
            upper_tick = math.floor(max_power / 100) * 0.1
            formatted_upper_tick = f"{upper_tick:.1f}"
            if upper_tick < 0:
                upper_tick = 0
                formatted_upper_tick = "0"
            # Has to round up for lower tick (so it doesn't fall off the image)
            lower_tick = math.ceil(min_power / 100) * 0.1
            formatted_lower_tick = f"{lower_tick:.1f}"
            if lower_tick > 0:
                lower_tick = 0
                formatted_lower_tick = "0"
            return (
                lower_tick * 1000,
                upper_tick * 1000,
                formatted_lower_tick,
                formatted_upper_tick,
            )

        def draw_device_schedule(
            self,
            draw: ImageDraw.ImageDraw,
            min_power,
            max_power,
            plot_height: int,
            x_pixels: np.ndarray,
            df_timeline: pd.DataFrame,
            curr_time_pos: int,
        ):
            """
            Draw the measured and predicted power consumption of
            the devices in the plot.
            Also tile the area between the previous and current line
            with the icons of the devices.
            The minimum space required for an icon is 8 pixels
            (set by calculate_icon_positions).
            """
            prev_measured_power_array = np.zeros(curr_time_pos + 1)
            prev_predicted_power_array = np.zeros(len(x_pixels) - curr_time_pos - 1)
            # initialize prev line with zero line
            prev_measured_points = power_array_to_points(
                prev_measured_power_array,
                min_power,
                max_power,
                plot_height,
                x_pixels[: curr_time_pos + 1],
            )
            prev_predicted_points = power_array_to_points(
                prev_predicted_power_array,
                min_power,
                max_power,
                plot_height,
                x_pixels[curr_time_pos + 1 :],
            )
            for device in self.devices:
                if not isinstance(device, SocketDevice):
                    # Note: The situation is more complex when considering batteries.
                    # Since these can provide power as well as consume power.
                    # A possible solution would be to track negative and
                    # positive power seperately
                    continue
                power_array = df_timeline[
                    TimelineColNames.measured_power_consumption(device)
                ].to_numpy()

                sliced_measured_power_array = power_array[: curr_time_pos + 1]
                stacked_measured_power_array = (
                    prev_measured_power_array + sliced_measured_power_array
                )
                prev_measured_power_array = stacked_measured_power_array
                measured_power_points = power_array_to_points(
                    stacked_measured_power_array,
                    min_power,
                    max_power,
                    plot_height,
                    x_pixels[: curr_time_pos + 1],
                )
                draw_line_segments(
                    draw, measured_power_points, fill=self.epd.GRAY3, width=1
                )

                predicted_power_array = df_timeline[
                    TimelineColNames.predicted_power_consumption(device)
                ].to_numpy()
                sliced_predicted_power_array = predicted_power_array[
                    curr_time_pos + 1 :
                ]
                stacked_predicted_power_array = (
                    sliced_predicted_power_array + prev_predicted_power_array
                )
                prev_predicted_power_array = stacked_predicted_power_array
                predicted_power_points = power_array_to_points(
                    stacked_predicted_power_array,
                    min_power,
                    max_power,
                    plot_height,
                    x_pixels[curr_time_pos + 1 :],
                )
                draw_line_segments(
                    draw, predicted_power_points, fill=self.epd.GRAY2, width=1
                )
                # Tile the space between the previous and current line
                positions_and_sizes_predicted_power = calculate_icon_positions(
                    predicted_power_points, prev_predicted_points, init_icon_size=32
                )
                positions_and_sizes_measured_power = calculate_icon_positions(
                    measured_power_points, prev_measured_points, init_icon_size=32
                )
                self.tile_graph_with_icons(
                    device, positions_and_sizes_predicted_power, draw
                )
                self.tile_graph_with_icons(
                    device, positions_and_sizes_measured_power, draw
                )
                # Update prev lines with the predicted and measured points
                prev_measured_points = measured_power_points
                prev_predicted_points = predicted_power_points

        def tile_graph_with_icons(
            self,
            device: SocketDevice,
            positions_and_sizes: list[tuple[int, int, int]],
            draw: ImageDraw.ImageDraw,
        ):
            for x, y, icon_size in positions_and_sizes:
                if (device, icon_size) in self.all_resized_icons:
                    icon = self.all_resized_icons[(device, icon_size)]
                else:
                    icon = self.resized_icons[device].resize(
                        (icon_size, icon_size), Image.Resampling.LANCZOS
                    )
                    self.all_resized_icons[(device, icon_size)] = icon
                draw.bitmap((int(x), int(y)), icon, fill=None)

        def create_complete_image(
            self, df_timeline: pd.DataFrame | None, curr_timeindex: datetime | None
        ) -> Image.Image:
            logger = self.logger
            epd = self.epd
            font = self.font_large
            L_image = Image.new("L", (epd.width, epd.height), 255)
            draw = ImageDraw.Draw(L_image)

            for device in self.devices:
                x, y = self.positions[device]
                icon = self.resized_icons[device]
                # draw text
                percentage = round(device.energy_stored / device.energy_capacity * 100)
                text = f"{percentage}%"
                text_w, text_h = get_text_width_and_height(text, font)
                text_x = x + math.ceil((self.max_text_width - text_w) / 2)
                text_y = y + math.ceil((self.cell_height - text_h) / 2)
                draw.text((text_x, text_y), text, font=font, fill=0)

                # add icon
                icon_x = x + self.max_text_width
                icon_y = y + math.ceil((self.cell_height - ICON_SIZE) / 2)
                L_image.paste(icon, (icon_x, icon_y), mask=icon)

            if df_timeline is None:
                logger.warning("Drawing plot skipped, since the df_timeline is None")
            else:
                logger.info("Drawing plot on E-paper display")
                if curr_timeindex is None:
                    logger.error("Current timeindex is None, so no plot will be drawn")
                else:
                    plot_image = self.create_image_full_plot(
                        df_timeline, curr_timeindex
                    )
                    L_image.paste(plot_image, (0, self.height_all_icons))
            return L_image

        def draw_full_update(
            self, df_timeline: pd.DataFrame | None, curr_timeindex: datetime | None
        ):
            try:
                logger = self.logger
                logger.info("Attempting full update")
                full_image = self.create_complete_image(df_timeline, curr_timeindex)
                epd = self.epd
                epd.Init_4Gray()
                epd.display_4Gray(epd.getbuffer_4Gray(full_image))
                logger.info("Sleep E-paper display")
                epd.sleep()

            except IOError as e:
                logger.error(f"Full draw failed with error: {e}")
            except asyncio.CancelledError:
                logger.info("Full draw cancelled")
                epd4in2_V2.epdconfig.module_exit(cleanup=True)
                raise

        def clear_sleep_display(self):
            try:
                logger = self.logger
                logger.info("Clearing E-paper display")
                epd = self.epd
                epd.init()
                epd.Clear()
                logger.info("Sleep E-paper display")
                epd.sleep()

            except IOError as e:
                logger.error(f"Clear and sleep E-paper display failed with error: {e}")
            except asyncio.CancelledError:
                logger.info("Clear and sleep E-paper display cancelled")
                epd4in2_V2.epdconfig.module_exit(cleanup=True)
                raise


def get_text_width_and_height(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
):
    """
    Get the width and height of the given text with the given font.
    """
    text_bbox = font.getbbox(text)
    text_w = math.ceil(text_bbox[2] - text_bbox[0])
    text_h = math.ceil(text_bbox[3] - text_bbox[1])
    return text_w, text_h


def draw_line_segments(
    image_draw: ImageDraw.ImageDraw,
    line_segments: list[list[tuple]],
    fill: int,
    width: int,
):
    for points in line_segments:
        image_draw.line(points, fill=fill, width=width)


def power_array_to_points(
    power_array: np.ndarray,
    min_power,
    max_power,
    plot_height: int,
    x_pixels: np.ndarray,
) -> list[list[tuple]]:
    """
    Create line segments from the x_pixels and power array.
    A new line segment is started when nan is encountered.

    (E.g. due to missing data due to connection issues)
    """
    # Normalized to 0-1
    normalized_power = (power_array - min_power) / (max_power - min_power)
    y_pixels = (1 - normalized_power) * (plot_height - 1)
    # The assumption is that only y can have nan
    all_line_segments = []
    curr_line_segment = []
    for x, y in zip(x_pixels, y_pixels, strict=True):
        if np.isnan(y):
            if curr_line_segment:
                all_line_segments.append(curr_line_segment)
                curr_line_segment = []
        else:
            curr_line_segment.append((x, y))
    if curr_line_segment:
        all_line_segments.append(curr_line_segment)

    return all_line_segments


def calculate_icon_positions(
    line_segments_upper: list[list[tuple[int, int]]],
    line_segments_lower: list[list[tuple[int, int]]],
    init_icon_size: int = ICON_SIZE,
) -> list[tuple[int, int, int]]:
    """
    Calculate the positions of the icons with (x, y, size).
    The upper points are the upper points in the graph,
    and the lower points are the lower points in the graph.
    Since a pixel down ⬇ has a larger y in the display's coordinate system,
    it means that the upper points have smaller values than
    the lower points.
    """
    icon_positions_and_sizes = []
    min_icon_size = 8  # Minimum size to still be recognizable
    common_segments_upper, common_segments_lower = get_segments_upper_and_lower(
        line_segments_upper, line_segments_lower
    )
    for pixel_points_upper, pixel_points_lower in zip(
        common_segments_upper, common_segments_lower, strict=True
    ):
        if len(pixel_points_upper) <= 2 or len(pixel_points_lower) <= 2:
            return icon_positions_and_sizes
        dict_upper = dict(pixel_points_upper)
        dict_lower = dict(pixel_points_lower)

        common_x_set = set(dict_upper) & set(dict_lower)
        common_x_list = sorted(common_x_set)
        y_pixels_upper = [dict_upper[x] for x in common_x_list]
        y_pixels_lower = [dict_lower[x] for x in common_x_list]

        if any(
            y_pixels_lower[idx] < y_pixels_upper[idx]
            for idx in range(len(y_pixels_upper))
        ):
            error_msg = (
                "The lower power points should be lower than the upper power points. "
                "In other words, the y value of the lower points should be larger "
                "This is not the case for some of the given points."
            )
            raise ValueError(error_msg)
        # Find the maximum length of an icon possible based on width
        max_icon_size = min(init_icon_size, int(common_x_list[-1] - common_x_list[0]))
        icon_sizes = list(range(max_icon_size, min_icon_size - 1, -8))

        pixels_width_per_index = pixel_points_upper[1][0] - pixel_points_lower[0][0]
        skip_ranges = []
        for icon_size in icon_sizes:
            icon_indices = math.ceil(icon_size / pixels_width_per_index)
            start_window_index = 0
            end_window_index = icon_indices
            while end_window_index < len(pixel_points_upper):
                if any(
                    s < start_window_index < e or s < end_window_index < e
                    for s, e in skip_ranges
                ):
                    # if any(s <= start_window_index < e for s, e in skip_ranges):
                    start_window_index += 1
                    end_window_index += 1
                    continue
                upper_window = y_pixels_upper[start_window_index:end_window_index]
                lower_window = y_pixels_lower[start_window_index:end_window_index]
                # Calculate difference between
                # upper points (smaller values) and lower points (larger values)
                lowest_y_upper = max(upper_window)
                highest_y_lower = min(lower_window)
                if highest_y_lower - lowest_y_upper < icon_size:
                    # Icon does not fit here
                    # (only allow an icon to fit when upper is above lower)
                    start_window_index += 1
                    end_window_index += 1
                    continue
                # Icon fits here,
                # so save the position and size and continue 1 icon size later
                x_pos = pixel_points_upper[start_window_index][0]
                y_pos = lowest_y_upper
                icon_positions_and_sizes.append((x_pos, y_pos, icon_size))
                skip_ranges.append((start_window_index, end_window_index))
                start_window_index += icon_indices
                end_window_index += icon_indices
            # If this icon size cannot fit anywhere anymore,
            # attempt the next smaller icon size.
    return icon_positions_and_sizes


def get_segments_upper_and_lower(
    unfiltered_line_segments_upper: list[list[tuple[int, int]]],
    unfiltered_line_segments_lower: list[list[tuple[int, int]]],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """
    Given two lists containing line segments,
    returns the two lists, but only if both the upper and lower exist.
    """
    # Remove any segments that do not contain any elements.
    line_segments_upper = [
        segment for segment in unfiltered_line_segments_upper if segment
    ]
    line_segments_lower = [
        segment for segment in unfiltered_line_segments_lower if segment
    ]
    upper_indices = [0, 0]
    lower_indices = [0, 0]
    new_upper: list[list] = [[]]
    new_lower: list[list] = [[]]
    upper_changed_segment = False
    lower_changed_segment = False
    while not (
        (upper_indices[0] >= len(line_segments_upper))
        or (lower_indices[0] >= len(line_segments_lower))
    ):
        upper_segment = line_segments_upper[upper_indices[0]]
        lower_segment = line_segments_lower[lower_indices[0]]
        upper_tuple = upper_segment[upper_indices[1]]
        lower_tuple = lower_segment[lower_indices[1]]
        if lower_changed_segment or upper_changed_segment:
            new_upper.append([])
            new_lower.append([])
            upper_changed_segment = False
            lower_changed_segment = False
        if upper_tuple[0] == lower_tuple[0]:
            new_upper[-1].append(upper_tuple)
            new_lower[-1].append(lower_tuple)
            lower_changed_segment = increase_indices(lower_indices, len(lower_segment))
            upper_changed_segment = increase_indices(upper_indices, len(upper_segment))
        elif upper_tuple[0] > lower_tuple[0]:
            # Data missing in upper -->
            # Skip a lower tuple
            lower_changed_segment = increase_indices(lower_indices, len(lower_segment))
        else:
            # Data missing in lower -->
            # Skip an upper tuple
            upper_changed_segment = increase_indices(upper_indices, len(upper_segment))
        # If one of the current tuples is the last in the segment
        # --> missing data has occured
        # --> go to the next segment
    return new_upper, new_lower


def increase_indices(indices: list[int], segment_length: int) -> bool:
    """
    Increases indices[1] with 1.
    Unless it would go out of bounds (determined by segment length).
    Then indices[0] increases with 1 and indices[1] is set to 0.
    Returns True if the increase is for indices[0]
    """
    if indices[1] + 1 >= segment_length:
        indices[0] += 1
        indices[1] = 0
        return True
    indices[1] += 1
    return False
