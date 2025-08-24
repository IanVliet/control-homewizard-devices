# homewizard_project
Python project with the purpose of controlling devices from homewizard. Specifically to schedule the devices during the day based on available power (measured with P1 meter) and predicted available solar power (quartz-solar-forecast). This project focuses on the scheduling of devices such as batteries and boilers. More generally devices that store energy and can thus be fully charged eventually. The devices can be either needed or optional, where needed devices should be fully charged once a day and optional devices are only charged when there is enough power available.

## Scheduling algorithm
The needed devices are scheduled during the day first in the order of `"priority"` and afterwards in the order of `"max_power_usage"`. 
They algorithm consists of the following main steps:
1. Schedule the device at this time if there is more power available than the device's `"max_power_usage"`.
2. Schedule the device at this time if the combination of a battery and available power can provide more power than `"max_power_usage"`.
3. Schedule the device at the times with the predicted largest amount of available power.

Optional devices only schedule with options 1.

## Controller
The `DeviceController` class controls the collection of data from the devices, the setting of states of the devices, acquiring the solar power prediction and deciding when each of these steps occurs. It collects data from the devices by sending measurement requests via the homewizard API (V1 for sockets and P1s and V2 for the battery). At the moment of writing this code the python package that includes V2 from homewizard requires python 3.12 atleast, and since the raspberry pi has python 3.11. I choose to write a simplified wrapper ('hwe_v2_wrapper') that includes getting info from the battery via measurements. So the wrapper does not include funtionality to update the state of the battery or anything else.

The controller also obtains a combined power forecast for the solar panels defined in `config_devices.json`. It does this upon start of the app and every day at 06:00 (Europe/Amsterdam, but can be changed in the code). The forecast is sliced from the current timestamp until 06:00 the next day. The forecast is also updated with the currently available power measured by the devices. The resulting difference at the current timestamp is used as a constant for the entire forecast.

The controller periodically (every 30s) requests data from the devices in `config_devices.json`. The data from sockets gives the power usage and state of the device. The the P1 meter combined with sockets and battery give the total power currently available (that is produced by the solar panels and is left over). The data and state is used to update the expected energy stored of each device. The scheduling algorithm is then used to determine which device should be turned at this moment.
