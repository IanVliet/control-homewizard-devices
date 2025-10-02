from zoneinfo import ZoneInfo

DELTA_T_SEC = 900  # 15 minutes in seconds
DELTA_T = DELTA_T_SEC / 3600  # Convert seconds to hours
PERIODIC_SLEEP_DURATION = 30  # Sleep duration for periodic tasks in seconds

# Below this ratio of inst power to full power the device is
# considered to be fully charged
IS_FULL_POWER_RATIO = 0.1

AGGREGATE_BATTERY = "aggregate_battery"

TZ = ZoneInfo("Europe/Amsterdam")

ICON_SIZE = 48
FONT_SIZE = 16
