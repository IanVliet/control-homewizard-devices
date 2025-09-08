import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from control_homewizard_devices.device_classes import SocketDevice, Battery
from control_homewizard_devices.schedule_devices import ColNames
from control_homewizard_devices.utils import is_raspberry_pi
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import io

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
            bbox = self.font.getbbox("100%")
            max_text_width = math.ceil(bbox[2] - bbox[0])
            text_height = math.ceil(bbox[3] - bbox[1])
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

        def draw_full_update(self, df_schedule: pd.DataFrame):
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
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = math.ceil(text_bbox[2] - text_bbox[0])
                    text_h = math.ceil(text_bbox[3] - text_bbox[1])
                    text_x = x + math.ceil((self.max_text_width - text_w) / 2)
                    text_y = y + math.ceil((self.cell_height - text_h) / 2)
                    draw.text((text_x, text_y), text, font=font, fill=0)

                    # add icon
                    icon_x = x + self.max_text_width
                    icon_y = y + math.ceil((self.cell_height - ICON_SIZE) / 2)
                    L_image.paste(icon, (icon_x, icon_y), mask=icon)

                # TODO: Draw the pandas dataframe via a plot
                if df_schedule is None:
                    logger.warning(
                        "Drawing plot skipped, since the df_schedule is None"
                    )
                else:
                    logger.info("Drawing plot on E-paper display")
                    # --- Matplotlib Figure ---
                    canvas_width, canvas_height = (
                        epd.width,
                        epd.height - self.height_all_icons,
                    )
                    dpi = 100  # dots per inch
                    fig_width, fig_height = (
                        canvas_width / dpi,
                        canvas_height / dpi,
                    )  # inches

                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

                    # Plot two columns
                    df_schedule.plot(ax=ax, y=[ColNames.POWER_W], color=["gray"])

                    # Formatting
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Value")
                    # ax.set_title("Daily Sensor Readings")
                    ax.grid(False)
                    ax.set_facecolor("white")
                    fig.patch.set_facecolor("white")
                    # Reduce x-axis labels if too many
                    ax.xaxis.set_major_locator(MaxNLocator(8))

                    # Remove extra padding
                    fig.tight_layout()

                    # --- Save figure to in-memory PNG ---
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    plt.close(fig)  # free memory

                    # --- Load with Pillow and convert to L mode ---
                    img = Image.open(buf)
                    img_l = img.convert("L")  # grayscale
                    L_image.paste(img_l, (0, self.height_all_icons), mask=img_l)
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
