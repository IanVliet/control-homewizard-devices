from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
# from plotly import graph_objects as go

if __name__ == "__main__":
    with open("config/config_devices.json", "r") as config_file:
        config_data = json.load(config_file)
    solar_panels_data = config_data["solar panels"]
    list_of_predictions = []
    today_date = datetime.today()
    for solar_panel_data in solar_panels_data:
        name = solar_panel_data["name"]
        params = solar_panel_data["quartz_parameters"]
        site = PVSite(**params)

        df_forecast = run_forecast(site=site, ts=today_date, nwp_source="icon")
        df_forecast.rename(columns={"power_kw": f"power_kw ({name})"}, inplace=True)
        list_of_predictions.append(df_forecast)
    df = pd.concat(list_of_predictions, axis=1)
    df.to_csv("data/solar_prediction.csv")
    fig = px.line(
        df,
        x=df.index,
        y=df.columns,
        title="Solar Power Forecast",
    )
    fig.show()
