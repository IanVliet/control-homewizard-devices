import asyncio
import signal
import logging

# from homewizard_energy import HomeWizardEnergyV1
from homewizard_energy import HomeWizardEnergy
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    P1Device,
    Battery,
)
import json

DEVICE_TYPE_MAP = {"HWE-SKT": SocketDevice, "HWE-P1": P1Device, "HWE-BAT": Battery}


class GracefulKiller:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.shutdown_event = asyncio.Event()

        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, self.trigger_shutdown)
            loop.add_signal_handler(signal.SIGTERM, self.trigger_shutdown)
            self.logger.info("Signal handlers registered (SIGINT, SIGTERM).")
        except (NotImplementedError, RuntimeError) as e:
            self.logger.warning(
                f"Signal handlers not supported: {e} \nLooking for KeyboardInterrupt"
            )

    def trigger_shutdown(self):
        if not self.shutdown_event.is_set():
            self.logger.info("Cancelling loop and shutting down...")
            self.shutdown_event.set()

    async def wait_for_shutdown(self):
        try:
            await self.shutdown_event.wait()
        except asyncio.CancelledError:
            self.logger.info("KeyboardInterrupt caught. Triggering shutdown...")
            self.trigger_shutdown()


# Setting up the logger in the main function
def setup_logger(log_level):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=log_level,  # Log messages of level log_level or higher
        format="%(asctime)s [%(levelname)s] - %(message)s",  # Log format with timestamp
        handlers=[
            logging.StreamHandler(),  # Logs to the console
            logging.FileHandler("app.log"),  # Log to file 'app.log'
        ],
    )
    return logger


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
