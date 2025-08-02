from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime
import json
from plotly import graph_objects as go

if __name__ == "__main__":
    with open("config_devices.json", "r") as config_file:
        config_data = json.load(config_file)
    solar_panels_data = config_data["solar panels"]
    for solar_panel_data in solar_panels_data:
        site = PVSite(
            latitude=solar_panel_data["latitude"],
            longitude=solar_panel_data["longitude"],
            capacity_kwp=solar_panel_data["total peak power"],
            tilt=solar_panel_data["tilt"],
            orientation=solar_panel_data["orientation"],
        )

        predictions_df = run_forecast(site=site, ts=datetime.today(), nwp_source="icon")
        predictions_df.to_csv("../data/solar_prediction.csv")
        predictions_df.to_parquet("../data/solar_prediction.parquet")
        fig = go.Figure(go.Scatter(y=predictions_df["power_kw"]))
        fig.show()
        break
