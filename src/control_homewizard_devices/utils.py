import logging
from control_homewizard_devices.device_classes import (
    CompleteDevice,
    SocketDevice,
    P1Device,
    Battery,
)
import json

DEVICE_TYPE_MAP = {"HWE-SKT": SocketDevice, "HWE-P1": P1Device, "HWE-BAT": Battery}


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
