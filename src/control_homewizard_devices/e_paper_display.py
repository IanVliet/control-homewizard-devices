import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from control_homewizard_devices.device_classes import SocketDevice, Battery
from control_homewizard_devices.schedule_devices import ColNames
from control_homewizard_devices.utils import is_raspberry_pi, TimelineColNames
from control_homewizard_devices.constants import ICON_SIZE, FONT_SIZE
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
                self.font = ImageFont.load_default(size=FONT_SIZE)
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

        def _get_text_width_and_height(
            self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        ):
            text_bbox = font.getbbox(text)
            text_w = math.ceil(text_bbox[2] - text_bbox[0])
            text_h = math.ceil(text_bbox[3] - text_bbox[1])
            return text_w, text_h

        def grid_positions(self):
            self.logger.info(
                "Calculating positions for cells containing charge percentages "
                "and icons"
            )
            max_text_width, text_height = self._get_text_width_and_height(
                "100%", self.font
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

            # Draw label power for y-axis
            # TODO: Draw label in kw.
            y_label_text = ColNames.POWER_W
            y_label_w, y_label_h = self._get_text_width_and_height(
                y_label_text, self.font
            )
            y_label_image = Image.new("L", (y_label_w, y_label_h), epd.GRAY1)
            y_label_draw = ImageDraw.Draw(y_label_image)
            y_label_draw.text((0, 0), y_label_text, fill=epd.GRAY4)
            # Rotate the y-label and add it to the plot
            rotated_y_label = y_label_image.rotate(90, expand=True)
            x_pos_y_label = 0
            y_pos_y_label = math.ceil((canvas_height - y_label_w) / 2)
            # TODO: Ensure the y-label does not overlap with the ticks.
            plot_image.paste(rotated_y_label, (x_pos_y_label, y_pos_y_label))

            # Calculate position x position and space needed for x-axis label
            x_label_text = "Time"
            x_label_w, x_label_h = self._get_text_width_and_height(
                x_label_text, self.font
            )
            # Determine the ticks and labels for the x-axis
            hour_format = "%H:%M"
            start_time_tick = df_timeline.index[0]
            end_time_tick = df_timeline.index[-1]
            formatted_start_time = start_time_tick.strftime(hour_format)
            start_time_w, start_time_h = self._get_text_width_and_height(
                formatted_start_time, self.font
            )
            formatted_end_time = end_time_tick.strftime(hour_format)
            end_time_w, end_time_h = self._get_text_width_and_height(
                formatted_end_time, self.font
            )

            self.logger.debug("Formating current time label")
            formatted_current_time = curr_timeindex.strftime(hour_format)
            curr_time_label_w, curr_time_label_h = self._get_text_width_and_height(
                formatted_current_time, self.font
            )
            # Calculate max height needed for different x-axis ticks and labels
            # TODO: Take the lower tick of the y-axis into account.
            max_x_label_h = max(
                x_label_h,
                start_time_h,
                end_time_h,
                curr_time_label_h,
            )
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
                self._get_text_width_and_height(formatted_lower_tick_y, self.font)
            )
            formatted_upper_tick_y_w, formatted_upper_tick_y_h = (
                self._get_text_width_and_height(formatted_upper_tick_y, self.font)
            )

            # Calculate max width needed for y-axis
            max_y_label_w = max(
                y_label_h,
                math.ceil(start_time_w / 2),
                formatted_lower_tick_y_w,
                formatted_upper_tick_y_w,
            )
            # --- Draw data ---
            top_left_point = (max_y_label_w, 0)
            # Calculate the area for the plot
            plot_width, plot_height = (
                canvas_width - max_y_label_w,
                canvas_height - max_x_label_h,
            )
            # Convert the time series to pixel positions
            num_datapoints = len(df_timeline.index)
            x_pixels = np.linspace(0, plot_width - 1, num_datapoints)
            # Initialize the data image with white background
            data_image = Image.new("L", (plot_width, plot_height), color=epd.GRAY1)
            data_draw = ImageDraw.Draw(data_image)

            self.logger.debug("Current timeindex: %s", curr_timeindex)
            # Get the position of the current time index
            if curr_timeindex in df_timeline.index:
                curr_time_pos = int(df_timeline.index.get_indexer([curr_timeindex])[0])
            else:
                curr_time_pos = None

            if curr_time_pos is not None:
                measured_power = df_timeline[TimelineColNames.MEASURED_POWER].to_numpy()
                notnan_mask = ~np.isnan(measured_power)
                notnan_pos = np.where(notnan_mask)[0][0] if notnan_mask.any() else None
                # Should go from the first not-nan value to the current time position
                if notnan_pos is None:
                    self.logger.debug(
                        "All measured power values are NaN, "
                        "skipping drawing measured power"
                    )
                else:
                    self.logger.debug(
                        "Drawing measured power from index: %s up to index: %s",
                        notnan_pos,
                        curr_time_pos,
                    )
                    if np.isnan(measured_power[notnan_pos : curr_time_pos + 1]).any():
                        error_msg = (
                            "Measured power contains NaN values between "
                            "the first not-NaN value and the current time position. "
                            "Skipping..."
                        )
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    measured_power_points = self.power_array_to_points(
                        measured_power[notnan_pos : curr_time_pos + 1],
                        min_power,
                        max_power,
                        plot_height,
                        x_pixels[notnan_pos : curr_time_pos + 1],
                    )
                    data_draw.line(measured_power_points, fill=epd.GRAY4, width=2)
                    self.logger.debug("Drawing scheduled devices")
                    # TODO: Draw rectangles or something for the scheduled devices.
                    # TODO: Try an icon of each device under the line.
                    # TODO: Consider what to do when the icon does not fit
                    self.draw_device_schedule(
                        data_draw,
                        min_power,
                        max_power,
                        plot_height,
                        x_pixels,
                        df_timeline,
                        notnan_mask,
                        notnan_pos,
                        curr_time_pos,
                    )

            # Draw a vertical line for the current time
            if curr_time_pos is not None and 0 <= curr_time_pos < num_datapoints:
                self.logger.debug(
                    "Drawing current time line at index: %s", curr_time_pos
                )
                curr_time_x = x_pixels[curr_time_pos]
                data_draw.line(
                    [(curr_time_x, 0), (curr_time_x, plot_height - 1)],
                    fill=epd.GRAY4,
                )
                self.logger.debug("Drawing current time label")
                x_pos_current_time = (
                    curr_time_x - curr_time_label_w // 2 + max_y_label_w
                )
                plot_draw.text(
                    (
                        x_pos_current_time,
                        canvas_height - max_x_label_h,
                    ),
                    formatted_current_time,
                    fill=epd.GRAY4,
                )

            # Draw line for predicted power
            self.logger.debug("Drawing predicted power line")
            predicted_power = df_timeline[TimelineColNames.PREDICTED_POWER].to_numpy()
            predicted_power_points = self.power_array_to_points(
                predicted_power, min_power, max_power, plot_height, x_pixels
            )
            data_draw.line(predicted_power_points, fill=epd.GRAY2, width=2)

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
            y_pos_x_label = canvas_height - max_x_label_h
            try:
                x_pos_x_label = self.get_position_label_avoid_overlap(
                    x_pos_current_time, curr_time_label_w, x_label_w, canvas_width
                )
            except NameError:
                self.logger.error(
                    "No position for current time tick, "
                    "so no overlap between tick for current time and x-label possible"
                )
            plot_draw.text((x_pos_x_label, y_pos_x_label), x_label_text, fill=epd.GRAY4)
            # Draw the start and end tick on the x-axis.
            # Only in case they do not overlap with the current tick.
            if x_pos_current_time > max_y_label_w + math.ceil(start_time_w / 2):
                plot_draw.text(
                    (max_y_label_w - start_time_w // 2, y_pos_x_label),
                    formatted_start_time,
                    fill=epd.GRAY4,
                )
            if x_pos_current_time < canvas_width - end_time_w:
                # TODO: Ensure the tick does not fall off the display.
                plot_draw.text(
                    (canvas_width - end_time_w // 2, y_pos_x_label),
                    formatted_end_time,
                    fill=epd.GRAY4,
                )

            # TODO: Group relevant codes together into logical positions and functions
            # TODO: Ensure the ticks are drawn if they do not overlap with zero label.
            # Draw the max tick on the y-axis
            plot_draw.text(
                (
                    0,
                    self.power_value_to_y_pixel(
                        upper_tick_y, min_power, max_power, plot_height
                    )
                    - formatted_upper_tick_y_h // 2,
                ),
                formatted_upper_tick_y,
                fill=epd.GRAY4,
            )
            # Draw the min tick on the y-axis
            plot_draw.text(
                (
                    0,
                    self.power_value_to_y_pixel(
                        lower_tick_y, min_power, max_power, plot_height
                    )
                    - formatted_lower_tick_y_h // 2,
                ),
                formatted_lower_tick_y,
                fill=epd.GRAY4,
            )
            # Draw tick for 0 line.

            return plot_image

        def calculate_min_max_power(self, df_timeline: pd.DataFrame):
            predicted_power = df_timeline[TimelineColNames.PREDICTED_POWER].to_numpy()
            measured_power = df_timeline[TimelineColNames.MEASURED_POWER].to_numpy()
            # Get the maximum of the predicted and measured power
            # TODO: In case the scheduled devices are included into the graph -->
            # take the devices into accounting when calculating the min and max
            max_measured = np.nanmax(measured_power, initial=0)
            max_predicted = np.nanmax(predicted_power, initial=0)
            self.logger.debug(
                "Max measured power: %s, Max predicted power: %s",
                max_measured,
                max_predicted,
            )
            max_stacked_power = np.array(
                [
                    max_predicted,
                    max_measured,
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
            min_stacked_power = np.array(
                [
                    min_predicted,
                    min_measured,
                    0,
                ]
            )  # Ensure min power is atleast 0
            min_power = np.nanmin(min_stacked_power)
            return min_power, max_power

        def power_array_to_points(
            self,
            power_array: np.ndarray,
            min_power,
            max_power,
            plot_height: int,
            x_pixels: np.ndarray,
        ):
            # Normalized to 0-1
            normalized_power = (power_array - min_power) / (max_power - min_power)
            y_pixels = (1 - normalized_power) * (plot_height - 1)
            points = list(zip(x_pixels, y_pixels, strict=True))
            return points

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
        ):
            init_label_start_pos = (total_length - label_length) // 2
            init_label_end_pos = init_label_start_pos + label_length
            tick_end_pos = tick_start_pos + tick_length
            diff_tick_end_to_label = init_label_start_pos - tick_end_pos
            diff_label_end_to_label = tick_start_pos - init_label_end_pos
            # Two cases for overlap:
            # 1. Tick is left of center and would overlap with label
            # (End of tick is past the start of the label)
            # --> move label to the right
            # 2. Tick is right of center and would overlap with label
            # (End of label is past the start of the tick)
            # --> move label to the left
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
            notnan_mask: np.ndarray,
            notnan_pos: int,
            curr_time_pos: int,
        ):
            prev_measured_power_array = np.zeros(curr_time_pos + 1 - notnan_pos)
            prev_predicted_power_array = np.zeros(len(x_pixels) - curr_time_pos - 1)
            # TODO: initialize prev line with zero line
            prev_measured_points = self.power_array_to_points(
                np.zeros(len(prev_measured_power_array)),
                min_power,
                max_power,
                plot_height,
                x_pixels[notnan_pos : curr_time_pos + 1],
            )
            prev_predicted_points = self.power_array_to_points(
                np.zeros(len(prev_predicted_power_array)),
                min_power,
                max_power,
                plot_height,
                x_pixels[curr_time_pos + 1 :],
            )
            for device in self.devices:
                if not isinstance(device, SocketDevice):
                    continue
                power_array = df_timeline[
                    TimelineColNames.measured_power_consumption(device)
                ].to_numpy()
                # Notnan mask should match the measured power array
                if not np.array_equal(notnan_mask, ~np.isnan(power_array)):
                    error_msg = (
                        "The not-nan mask does not match the measured power array "
                        f"for device {device.device_name}. "
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                sliced_measured_power_array = power_array[
                    notnan_pos : curr_time_pos + 1
                ]
                stacked_measured_power_array = (
                    prev_measured_power_array + sliced_measured_power_array
                )
                prev_measured_power_array = stacked_measured_power_array
                measured_power_points = self.power_array_to_points(
                    stacked_measured_power_array,
                    min_power,
                    max_power,
                    plot_height,
                    x_pixels[notnan_pos : curr_time_pos + 1],
                )
                draw.line(measured_power_points, fill=self.epd.GRAY3, width=1)

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
                predicted_power_points = self.power_array_to_points(
                    stacked_predicted_power_array,
                    min_power,
                    max_power,
                    plot_height,
                    x_pixels[curr_time_pos + 1 :],
                )
                draw.line(predicted_power_points, fill=self.epd.GRAY2, width=1)
                # TODO: Tile the space brtween the previous and current line
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
            device: SocketDevice | Battery,
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

        def draw_full_update(
            self, df_timeline: pd.DataFrame | None, curr_timeindex: datetime | None
        ):
            try:
                logger = self.logger
                logger.info("Attempting full update")
                epd = self.epd
                epd.init()
                # TODO: Test/Consider whether clear is needed.
                epd.Clear()
                epd.Init_4Gray()
                font = self.font
                L_image = Image.new("L", (epd.width, epd.height), 255)
                draw = ImageDraw.Draw(L_image)

                for device in self.devices:
                    x, y = self.positions[device]
                    icon = self.resized_icons[device]
                    # draw text
                    percentage = round(
                        device.energy_stored / device.energy_capacity * 100
                    )
                    text = f"{percentage}%"
                    text_w, text_h = self._get_text_width_and_height(text, self.font)
                    text_x = x + math.ceil((self.max_text_width - text_w) / 2)
                    text_y = y + math.ceil((self.cell_height - text_h) / 2)
                    draw.text((text_x, text_y), text, font=font, fill=0)

                    # add icon
                    icon_x = x + self.max_text_width
                    icon_y = y + math.ceil((self.cell_height - ICON_SIZE) / 2)
                    L_image.paste(icon, (icon_x, icon_y), mask=icon)

                # TODO: Draw the pandas dataframe via a plot
                if df_timeline is None:
                    logger.warning(
                        "Drawing plot skipped, since the df_timeline is None"
                    )
                else:
                    logger.info("Drawing plot on E-paper display")
                    # TODO: Manually draw plot
                    if curr_timeindex is None:
                        logger.error(
                            "Current timeindex is None, so no plot will be drawn"
                        )
                    else:
                        plot_image = self.create_image_full_plot(
                            df_timeline, curr_timeindex
                        )
                        L_image.paste(plot_image, (0, self.height_all_icons))
                epd.display_4Gray(epd.getbuffer_4Gray(L_image))
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


def calculate_icon_positions(
    pixel_points_upper: list[tuple[int, int]],
    pixel_points_lower: list[tuple[int, int]],
    init_icon_size: int = ICON_SIZE,
) -> list[tuple[int, int, int]]:
    if len(pixel_points_upper) != len(pixel_points_lower):
        error_msg = (
            "The upper and lower power points lists should have the same length. "
            f"Got {len(pixel_points_upper)} and {len(pixel_points_lower)}"
        )
        raise ValueError(error_msg)
    if len(pixel_points_upper) <= 2:
        return []
    # TODO: Properly take into account that power can sometimes be negative,
    # and thus flip the upper and lower points.
    # Find the maximum length of an icon possible based on width
    max_icon_size = min(
        init_icon_size, int(pixel_points_upper[-1][0] - pixel_points_upper[0][0])
    )
    min_icon_size = 8  # Minimum size to still be recognizable
    icon_sizes = list(range(max_icon_size, min_icon_size - 1, -8))
    pixels_width_per_index = pixel_points_upper[1][0] - pixel_points_lower[0][0]
    icon_positions_and_sizes = []
    y_pixels_upper = [point[1] for point in pixel_points_upper]
    y_pixels_lower = [point[1] for point in pixel_points_lower]
    skip_ranges = []
    for icon_size in icon_sizes:
        # TODO: While looping through icon sizes from max to min:
        icon_indices = math.ceil(icon_size / pixels_width_per_index)
        start_window_index = 0
        end_window_index = icon_indices
        while end_window_index < len(pixel_points_upper):
            if any(s <= start_window_index < e for s, e in skip_ranges):
                start_window_index += 1
                end_window_index += 1
                continue
            upper_window = y_pixels_upper[start_window_index:end_window_index]
            lower_window = y_pixels_lower[start_window_index:end_window_index]
            # Calculate difference between
            # upper points (that have smaller values) and lower points (larger values)
            lowest_y_upper = max(upper_window)
            highest_y_lower = min(lower_window)
            if abs(lowest_y_upper - highest_y_lower) < icon_size:
                start_window_index += 1
                end_window_index += 1
                continue
            # Icon fits here,
            # so save the position and size and continue 1 pixel size later
            x_pos = pixel_points_upper[start_window_index][0]
            y_pos = lowest_y_upper
            icon_positions_and_sizes.append((x_pos, y_pos, icon_size))
            skip_ranges.append((start_window_index, end_window_index))
            start_window_index += icon_indices
            end_window_index += icon_indices
        # If this icon size cannot fit anywhere anymore,
        # continue with the next smaller icon size.
    return icon_positions_and_sizes
