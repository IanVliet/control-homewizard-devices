import asyncio
from homewizard_energy import HomeWizardEnergy

from control_homewizard_devices.utils import GracefulKiller, setup_logger, initialize_devices
from contextlib import AsyncExitStack
import logging
import sys
from control_homewizard_devices.device_classes import complete_device, socket_device, p1_device


def get_total_available_power(all_devices: list[complete_device]):
    return sum(power for power in (device.get_instantaneous_power() for device in all_devices) if power is not None)


def determine_socket_states(total_power, sorted_sockets: list[socket_device]):
    # available power is defined as - total power
    available_power = - total_power

    for socket in sorted_sockets:
        if socket.should_power_on(available_power):
            socket.updated_state = True
            available_power -= socket.max_power_usage
        else:
            socket.updated_state = False



async def main_loop(all_devices: list[complete_device], sorted_sockets: list[socket_device], logger: logging.Logger):
    try:
        while True:
            # TODO: Ensure that if one task fails (i.e. one device cannot do a measurement) 
            # the code continues with the other devices and sends some kind of message about the failed measurement
            async with asyncio.TaskGroup() as tg:
                for device in all_devices:
                    tg.create_task(device.perform_measurement(logger))
            logger.info(f"===== devices info gathered =====")

            total_power = get_total_available_power(all_devices)

            determine_socket_states(total_power, sorted_sockets)
            async with asyncio.TaskGroup() as tg:
                for socket in sorted_sockets:
                    tg.create_task(socket.update_power_state(logger))
            logger.info(f"===== socket states updated =====")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Loop cancelled gracefully...")
    finally:
        logger.info("Cleaning up before shutdown...")



async def main(all_devices: list[complete_device]):
    socket_devices = [device for device in all_devices if isinstance(device, socket_device)]
    sorted_sockets = sorted(socket_devices, key=lambda d: d.priority)

    logger = setup_logger(logging.INFO)
    killer = GracefulKiller(logger)
    async with AsyncExitStack() as stack:
        hwe_devices = []
        for device in all_devices:
            device.hwe_device = await stack.enter_async_context(HomeWizardEnergy(host=device.ip_address))
            hwe_devices.append(device.hwe_device)

        main_task = asyncio.create_task(main_loop(all_devices, sorted_sockets, logger))

        # wait for keyboardInterrupt to cancel the main task
        await killer.wait_for_shutdown()

        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            logger.warning("CancelledError propogated")
        except KeyboardInterrupt:
            logger.error("Unexpected KeyboardInterrupt caught during task cancellation!")

        logger.info("Shutdown complete")
        sys.exit(130)

def entrypoint():
    all_devices = initialize_devices("./config_devices.json")
    asyncio.run(main(all_devices))

if  __name__ == "__main__":
    entrypoint()
