import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from control_homewizard_devices.device_classes import SocketDevice, Battery
from control_homewizard_devices.schedule_devices import ColNames
from control_homewizard_devices.utils import is_raspberry_pi, TimelineColNames
import pandas as pd
import numpy as np

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

    ICON_SIZE = 48
    FONT_SIZE = 16

    class DrawDisplay:
        def __init__(
            self, devices: list[SocketDevice | Battery], logger: logging.Logger
        ) -> None:
            self.devices = devices
            self.resized_icons = self.get_resized_icons()
            self.logger = logger
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

        def create_image_full_plot(self, df_timeline: pd.DataFrame) -> Image.Image:
            epd = self.epd
            canvas_width, canvas_height = (
                epd.width,
                epd.height - self.height_all_icons,
            )
            plot_image = Image.new("L", (canvas_width, canvas_height), 255)
            plot_draw = ImageDraw.Draw(plot_image)

            # Draw label time for x-axis
            x_label_text = "Time"
            x_label_w, x_label_h = self._get_text_width_and_height(
                x_label_text, self.font
            )
            x_pos_x_label = math.ceil((canvas_width - x_label_w) / 2)
            y_pos_x_label = canvas_height - x_label_h
            plot_draw.text((x_pos_x_label, y_pos_x_label), x_label_text, fill=epd.GRAY4)

            # Draw label power for x-axis
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
            plot_image.paste(rotated_y_label, (x_pos_y_label, y_pos_y_label))

            # Draw x-axis line
            top_left_point = (y_label_h, 0)
            bottom_left_point = (y_label_h, canvas_height - x_label_h)
            bottom_right_point = (canvas_width, canvas_height - x_label_h)
            plot_draw.line(
                (top_left_point, bottom_left_point),
                fill=epd.GRAY4,
            )
            # TODO: Draw the 0 line either on the bottom or
            # if the minimum of the measured power and predicted power is less than 0
            # draw the line at the pixel height that is closest to 0.
            plot_draw.line(
                (bottom_left_point, bottom_right_point),
                fill=epd.GRAY4,
            )
            # Draw predicted power line
            plot_width, plot_height = (
                canvas_width - y_label_h,
                canvas_height - x_label_h,
            )
            num_datapoints = len(df_timeline.index)
            predicted_power = df_timeline[TimelineColNames.PREDICTED_POWER].to_numpy()
            measured_power = df_timeline[TimelineColNames.MEASURED_POWER].to_numpy()
            # Get the maximum of the predicted and measured power
            # TODO: In case the scheduled devices are included into the graph -->
            # take the devices into accounting when calculating the min and max
            max_power = np.nanmax([predicted_power, measured_power])
            min_power = np.nanmin([predicted_power, measured_power])
            # Normalized to 0-1
            normalized_predicted_power = (predicted_power - min_power) / (
                max_power - min_power
            )
            x_pixels = np.linspace(0, plot_width - 1, num_datapoints)
            y_pixels = (1 - normalized_predicted_power) * (plot_height - 1)
            data_image = Image.new("L", (plot_width, plot_height), color=epd.GRAY1)
            data_draw = ImageDraw.Draw(data_image)

            # TODO: Draw the measured power up to the current time
            # TODO: Draw a vertical line for the current time
            points = list(zip(x_pixels, y_pixels, strict=False))
            data_draw.line(points, fill=epd.GRAY4)

            plot_image.paste(data_image, top_left_point)

            # Draw measured power line (only be available up to the current timestep)
            return plot_image

        def draw_full_update(self, df_timeline: pd.DataFrame | None):
            try:
                logger = self.logger
                logger.info("Attempting full update")
                epd = self.epd
                epd.init()
                epd.Clear()
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
                    plot_image = self.create_image_full_plot(df_timeline)
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
