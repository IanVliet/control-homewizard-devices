# homewizard_project
Python project with the purpose of controlling devices from [Homewizard](https://www.homewizard.com/). Specifically to schedule the devices based on available power (measured with P1 meter) and predicted available solar power ([quartz-solar-forecast](https://github.com/openclimatefix/open-source-quartz-solar-forecast)). This project focuses on scheduling devices that store energy such as batteries and boilers. The devices can be either needed or optional (set by the boolean flag `"daily_need"`). Needed devices are fully charged once a day and optional devices are only charged when there is enough power available. The created schedule attempts to get close to maximizing the use of the available power without drawing power from the grid unnecessarily. Only for needed devices can the schedule draw from a homewizard battery and the grid. An example is a kitchen boiler which should still provide hot water even if there is not enough solar power in the winter or on cloudy days thus drawing power from the grid/battery.

## Scheduling algorithm
The needed devices are scheduled during the day first in the order of `"priority"` and afterwards in the order of `"max_power_usage"`. 
The algorithm consists of the following main steps for needed devices:
1. Schedule the device at this time if there is more power available than the device's `"max_power_usage"`.
2. Schedule the device at this time if the combination of a battery and available power can provide more power than `"max_power_usage"`.
3. Schedule the device at the times with the predicted largest amount of available power.

Next the optional devices are similarly ordered and scheduled with option 1.

## Controller
The `DeviceController` class controls the collection of data from the devices, the setting of states of the devices, acquiring the solar power prediction and deciding when each of these steps occurs. It collects data from the devices by sending measurement requests via the homewizard API (V1 for sockets and P1s and V2 for the battery). 
### Note
The raspberry has python 3.11 and the existing python API for V2 from [python-homewizard-energy](https://github.com/homewizard/python-homewizard-energy) requires python 3.12 or higher (as of writing this). Therefore, I choose to slightly adjust existing code from [python-homewizard-energy](https://github.com/homewizard/python-homewizard-energy) into a simplified wrapper ('hwe_v2_wrapper'). The wrapper gets info from the battery via measurements only and does not include funtionality to update the state of the battery or do anything else.

The controller also obtains a combined power forecast for the solar panels defined in `config_devices.json`. It does this upon start of the app and every day at 06:00 (Europe/Amsterdam but can be changed in the code). The forecast is sliced from the current timestamp until 06:00 the next day. Before using the forecast in the scheduler, the forecast is updated with the currently available power measured by the devices. The resulting power difference at the current timestamp is used as a constant for the entire forecast.

The controller periodically (every 30s) requests data from the devices in `config_devices.json`. The measured data used from the sockets comes in the form of the power usage and state of the device. The power usages of the P1 meter, sockets and battery give the total power currently available. In other words, the power that is produced by the solar panels and is left over after other the consumption by other unregistered devices. The power usage and state are used to update the (expected) energy stored by each device. The scheduling algorithm is then used to determine which device should be turned at this moment.

## Config
The file that should be created in the `config` directory is `config_devices.json`. It must the relevant information for both the homewizard devices and the solar panels. See `example_config_devices.json` for an example. The properties can/should all be changed to properties of the specific device. The only exception is `"device_type"` since this should match the device type ("HWE-SKT", "HWE-P1" or "HWE-BAT"). To create a user for the homewizard API V2 follow the manual steps at [API V2 examples](https://api-documentation.homewizard.com/docs/v2/authorization#examples). 

## systemd service
To run the script as a systemd service (on a raspberry pi for example), there is an example systemd unit file named `hwe_control_script.service` in `example_files`. The file requires changing the [username] with the username on the computer and I recommend checking the validity of all paths.

Similarly, there are examples scripts for automatically updating the local repository on a computer with `deploy-check.service` and `deploy-check.timer`. Only `deploy-check.service` requires changing [username] with the username on the computer.

# Use the project directly
Assuming poetry is correctly installed on the computer and the github repository has been cloned; run the following in the main directory:
```
poetry install
```
For more information on poetry see [poetry basic usage](https://python-poetry.org/docs/basic-usage/). Additionally, a directory to store the logs should be created:
```
mkdir logs/
```

To run the main control_energy_systems.py, you can use the entrypoint defined in the pyproject.toml with:
```
poetry run myservice
```
To run the tests:
```
poetry install --with dev
poetry run pytest
```
(or just `pytest` instead of `poety run pytest` if the environment is activated ([activate poetry environment](https://python-poetry.org/docs/managing-environments#activating-the-environment)))

The file `scheduling_benchmark.py` can be run after `power_forecast.py` has been run and the filename of the created csv file matches the filename `DATA_FILENAME` in `scheduling_benchmark.py`.
