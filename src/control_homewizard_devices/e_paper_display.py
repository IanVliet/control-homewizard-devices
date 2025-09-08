import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from control_homewizard_devices.device_classes import SocketDevice, Battery

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
LIBS_PATH = os.path.join(REPO_ROOT, "libs")
waveshare_lib = os.path.join(
    LIBS_PATH, "e-Paper", "RaspberryPi_JetsonNano", "python", "lib"
)
icons_dir = os.path.join(REPO_ROOT, "icons")

if os.path.exists(waveshare_lib) and waveshare_lib not in sys.path:
    sys.path.append(waveshare_lib)

from waveshare_epd import epd4in2_V2

ICON_SIZE = 32
FONT_SIZE = 12


class DrawDisplay:
    def __init__(self, devices: list[SocketDevice | Battery]) -> None:
        self.devices = devices
        self.resized_icons = self.get_resized_icons()
        try:
            logging.info("Setting up E-paper display (epd) class")
            self.epd = epd4in2_V2.EPD()
            logging.info("Init and clear epd")
            self.epd.init()
            self.epd.Clear()
            self.font = ImageFont.load_default(size=FONT_SIZE)
            self.positions, cols, rows, self.height_all_icons = self.grid_positions()
        except IOError as e:
            logging.error(f"Setup epd failed with error: {e}")
        except asyncio.CancelledError:
            logging.info("Setup epd cancelled")
            epd4in2_V2.epdconfig.module_exit(cleanup=True)
            raise

    def get_resized_icons(self):
        resized_icons: dict[SocketDevice | Battery, Image.Image] = {}
        for device in self.devices:
            icon_path = os.path.join(
                icons_dir, device.device_name.replace(" ", "_"), ".bmp"
            )
            icon = Image.open(icon_path)
            resized = icon.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
            resized_icons[device] = resized
        return resized_icons

    def grid_positions(self):
        bbox = self.font.getbbox("100%")
        text_height = int(bbox[3] - bbox[1])
        # calculate grid for icon + text
        max_cols = self.epd.width // ICON_SIZE
        cols = min(max_cols, len(self.devices))

        total_icons_width = cols * ICON_SIZE
        if cols > 1:
            spacing = (self.epd.width - total_icons_width) // (cols - 1)
        else:
            spacing = 0

        rows = math.ceil(len(self.devices) / cols)
        total_height = rows * (ICON_SIZE + text_height)

        positions: dict[SocketDevice | Battery, tuple[int, int]] = {}

        for idx, device in enumerate(self.devices):
            row = idx // cols
            col = idx % cols
            x = col * (ICON_SIZE + spacing)
            y = row * (ICON_SIZE + text_height)
            positions[device] = (x, y)

        return positions, cols, rows, total_height

    async def draw_full_update(self):
        try:
            logging.info("Attempting full update")
            epd = self.epd
            font = self.font
            L_image = Image.new("L", (epd.width, epd.height), 255)
            draw = ImageDraw.Draw(L_image)

            for device in self.devices:
                x, y = self.positions[device]
                icon = self.resized_icons[device]
                # draw text
                percentage = round(device.energy_stored / device.energy_capacity * 100)
                text = f"{percentage}%"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = int(bbox[2] - bbox[0])
                text_height = int(bbox[3] - bbox[1])
                text_x = x + (ICON_SIZE - text_width) / 2
                text_y = y  # top of the cell

                draw.text((text_x, text_y), text, font=font, fill=0)

                # add icon
                icon_y = y + text_height
                L_image.paste(icon, (x, icon_y), mask=icon)
            epd.display_4Gray(epd.getbuffer_4Gray(L_image))
            await asyncio.sleep(5)
            logging.info("Clear and go to sleep epd")
            epd.init()
            epd.Clear()
            epd.sleep()

        except IOError as e:
            logging.error(f"Full draw failed with error: {e}")
        except asyncio.CancelledError:
            logging.info("Full draw cancelled")
            epd4in2_V2.epdconfig.module_exit(cleanup=True)
            raise
