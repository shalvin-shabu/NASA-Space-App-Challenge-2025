import requests
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

# Suppress warnings from statsmodels to keep the output clean
warnings.filterwarnings("ignore")

# ==============================
# 🔐 User Config (API Credentials)
# ==============================
AQICN_API_TOKEN = "adab0250428e88549d94621afbdaa2c1c36d00c5"
METEO_USER = "menon_prithvishankar"
METEO_PASS = "Q6bRXp5GWf1mLoLa7Jz1"

# --- Define the core pollutants we always want to find ---
REQUIRED_POLLUTANTS = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

# ==============================
# ⚙️ Helper Functions
# ==============================
def get_location_name(lat, lon):
    headers = {'User-Agent': 'AirQualityForecastApp/1.0'}
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    try:
        response = requests.get(url, headers=headers); response.raise_for_status(); data = response.json()
        address = data.get('address', {}); location = address.get('city', address.get('town', address.get('village', 'Unknown Location')))
        return location
    except Exception: return "Unknown Location"

def get_waqi_data(lat, lon, token):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    try:
        response = requests.get(url); response.raise_for_status(); data = response.json()
        return data["data"].get("iaqi", {}) if data.get("status") == "ok" else None
    except requests.exceptions.RequestException: return None

# --- NEW FALLBACK FUNCTION ---
def get_fallback_data(city_name, missing_pollutants, token):
    """
    Searches for nearby stations in the same city to find missing pollutant data.
    """
    print(f"Searching for missing pollutants in {city_name}...")
    found_data = {}
    search_url = f"https://api.waqi.info/search/?keyword={city_name}&token={token}"
    try:
        res = requests.get(search_url); res.raise_for_status(); search_results = res.json()
        if search_results.get("status") != "ok": return {}

        # Limit search to the top 5 most relevant stations
        for station in search_results.get("data", [])[:5]:
            station_uid = station.get("uid")
            station_name = station.get("station", {}).get("name", "N/A")
            if not missing_pollutants: break # Stop if we've found everything
            if station_uid:
                feed_url = f"https://api.waqi.info/feed/@{station_uid}/?token={token}"
                feed_res = requests.get(feed_url); feed_res.raise_for_status(); station_data = feed_res.json()
                if station_data.get("status") == "ok":
                    station_pollutants = station_data.get("data", {}).get("iaqi", {})
                    for pollutant in list(missing_pollutants): # Iterate over a copy
                        if pollutant in station_pollutants:
                            value = station_pollutants[pollutant]
                            value['source'] = station_name # Add source info
                            found_data[pollutant] = value
                            missing_pollutants.remove(pollutant)
                            print(f"  -> Found {pollutant.upper()} at station: {station_name}")
        return found_data
    except Exception as e:
        print(f"  -> Fallback search failed: {e}")
        return {}

def get_air_quality_data_meteomatics(lat, lon, pollutants, username, password):
    pollutant_mapping = {'pm25': 'pm2p5:ugm3', 'pm10': 'pm10:ugm3', 'o3': 'o3:ugm3','no2': 'no2:ugm3', 'so2': 'so2:ugm3', 'co': 'co:ugm3'}
    reverse_mapping = {v: k for k, v in pollutant_mapping.items()}
    meteomatics_params = [pollutant_mapping[p] for p in pollutants if p in pollutant_mapping]
    if not meteomatics_params: return pd.DataFrame()
    params_str = ",".join(meteomatics_params)
    end_time = dt.datetime.utcnow() + dt.timedelta(hours=24)
    start_time = dt.datetime.utcnow() - dt.timedelta(days=90)
    time_str = f"{start_time.isoformat(timespec='seconds')}Z--{end_time.isoformat(timespec='seconds')}Z:PT1H"
    location_str = f"{lat},{lon}"; url = f"https://api.meteomatics.com/{time_str}/{params_str}/{location_str}/json"
    print("Querying Meteomatics API for historical & forecast data...")
    try:
        response = requests.get(url, auth=(username, password)); response.raise_for_status(); data = response.json()
        if data.get('status') != 'OK' or not data.get('data'): return pd.DataFrame()
        all_data = {}
        for param_data in data['data']:
            param_name, dates = param_data['parameter'], param_data['coordinates'][0]['dates']
            values = [d['value'] for d in dates]; all_data[param_name] = values
        df = pd.DataFrame(all_data)
        df['datetime'] = [d['date'] for d in data['data'][0]['coordinates'][0]['dates']]
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True); df.rename(columns=reverse_mapping, inplace=True)
        print("✅ Fetched historical & API forecast data from Meteomatics.")
        return df
    except Exception: return pd.DataFrame()

def forecast_pollutant_sarima(df, pollutant, hours_ahead=24):
    df_filtered = df.dropna(subset=[pollutant]).copy()
    if len(df_filtered) < 50: print(f"Not enough historical data to forecast {pollutant.upper()} with SARIMA."); return None
    time_series = df_filtered.set_index('datetime')[pollutant].asfreq('H')
    order, seasonal_order = (1, 1, 1), (1, 1, 1, 24)
    try:
        model = ARIMA(time_series, order=order, seasonal_order=seasonal_order); model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=hours_ahead); forecast_df = forecast.summary_frame()
        future_time = [time_series.index[-1] + dt.timedelta(hours=i+1) for i in range(hours_ahead)]
        return pd.DataFrame({"datetime": future_time, f"{pollutant}_forecast": forecast_df['mean'].values, f"{pollutant}_lower_ci": forecast_df['mean_ci_lower'].values, f"{pollutant}_upper_ci": forecast_df['mean_ci_upper'].values})
    except Exception: return None

def plot_pollutant(historical_df, api_forecast_df, sarima_forecast_df, pollutant, city):
    pollutant_name = pollutant.upper(); title = f"{pollutant_name} Data & 24-Hour Forecast – {city}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["datetime"], y=historical_df[pollutant], mode='lines', name='Historical Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=api_forecast_df["datetime"], y=api_forecast_df[pollutant], mode='lines', name='API Forecast', line=dict(color='#87CEEB', dash='dash')))
    if sarima_forecast_df is not None:
        forecast_col, lower_ci_col, upper_ci_col = f"{pollutant}_forecast", f"{pollutant}_lower_ci", f"{pollutant}_upper_ci"
        fig.add_trace(go.Scatter(x=sarima_forecast_df["datetime"], y=sarima_forecast_df[forecast_col], mode='lines', name='Our SARIMA Forecast', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=pd.concat([sarima_forecast_df["datetime"], sarima_forecast_df["datetime"][::-1]]), y=pd.concat([sarima_forecast_df[upper_ci_col], sarima_forecast_df[lower_ci_col][::-1]]), fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name='SARIMA 95% Confidence'))
        zoom_start = api_forecast_df["datetime"].iloc[0] - dt.timedelta(hours=24) if not api_forecast_df.empty else historical_df["datetime"].iloc[-1]
        zoom_end, full_view_start, full_view_end = sarima_forecast_df["datetime"].iloc[-1] + dt.timedelta(hours=1), historical_df["datetime"].min(), sarima_forecast_df["datetime"].max()
    else:
        full_view_start, full_view_end = historical_df["datetime"].min(), api_forecast_df["datetime"].max() if not api_forecast_df.empty else historical_df["datetime"].max()
        zoom_start, zoom_end = (api_forecast_df["datetime"].iloc[0] - dt.timedelta(hours=24) if not api_forecast_df.empty else historical_df["datetime"].iloc[-1]), (api_forecast_df["datetime"].max() if not api_forecast_df.empty else historical_df["datetime"].max())
    fig.update_layout(title=title, xaxis_title="Time (UTC)", yaxis_title=f"{pollutant_name} (Concentration / AQI)", updatemenus=[dict(type="buttons", direction="right", x=0.57, xanchor="left", y=1.15, yanchor="top", showactive=True, buttons=list([dict(label="Full View", method="relayout", args=[{"xaxis.range": [full_view_start, full_view_end]}]), dict(label="Zoom to Forecast", method="relayout", args=[{"xaxis.range": [zoom_start, zoom_end]}])]))])
    fig.show()

def plot_realtime_only(pollutant, data, timestamp, city):
    pollutant_name = pollutant.upper()
    value = data.get('v')
    source = data.get('source') # Check for a source
    title_source = f"(Value from nearby station: {source})" if source else ""
    title = f"{pollutant_name} Real-Time Value – {city}{title_source}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[timestamp], y=[value], mode='markers', marker=dict(size=15, color='red'), name='Current Value'))
    fig.update_layout(title=title, xaxis_title="Time (UTC)", yaxis_title=f"{pollutant_name} (Concentration / AQI)", annotations=[dict(x=timestamp, y=value, text=f"Current: {value}", showarrow=True, arrowhead=1, ax=20, ay=-40)])
    fig.show()

def generate_alerts(df, pollutant, threshold):
    pollutant_name, forecast_col = pollutant.upper(), f"{pollutant}_forecast"
    high_values = df[df[forecast_col] > threshold]
    if not high_values.empty:
        max_val = high_values[forecast_col].max()
        print(f"🚨 ALERT! Unhealthy {pollutant_name} levels predicted. Max forecast: {max_val:.2f} (Threshold: {threshold})")

# ==============================
# ▶️ Main Execution Block
# ==============================
while True:
    try:
        lat_input = float(input("Enter the latitude (e.g., 34.0522): ")); lon_input = float(input("Enter the longitude (e.g., -118.2437): "))
        if -90 <= lat_input <= 90 and -180 <= lon_input <= 180: break
        else: print("❌ Invalid range.")
    except ValueError: print("❌ Invalid input.")

city_name = get_location_name(lat_input, lon_input)
print(f"\n▶️ Running analysis for {city_name} ({lat_input}, {lon_input})...")

# --- MODIFIED DATA GATHERING LOGIC ---
real_time_aq_data = get_waqi_data(lat_input, lon_input, AQICN_API_TOKEN)
if real_time_aq_data is None: real_time_aq_data = {}

# Check which required pollutants are missing from the initial call
missing_pollutants = [p for p in REQUIRED_POLLUTANTS if p not in real_time_aq_data]

if missing_pollutants:
    print(f"⚠️ Missing data for: {', '.join(p.upper() for p in missing_pollutants)}")
    fallback_data = get_fallback_data(city_name, missing_pollutants, AQICN_API_TOKEN)
    real_time_aq_data.update(fallback_data) # Add the found data to our main dictionary

# Proceed with the full list of pollutants we've gathered
if real_time_aq_data:
    available_pollutants = [p for p in REQUIRED_POLLUTANTS if p in real_time_aq_data]
    print(f"✅ Using data for pollutants: {', '.join(p.upper() for p in available_pollutants)}")
    
    combined_df = get_air_quality_data_meteomatics(lat_input, lon_input, available_pollutants, METEO_USER, METEO_PASS)
    now_utc = pd.Timestamp.utcnow()
    HEALTH_THRESHOLDS = {'pm25': 35, 'pm10': 75, 'o3': 70, 'no2': 100, 'so2': 75, 'co': 9000}

    for pollutant in REQUIRED_POLLUTANTS:
        if pollutant not in available_pollutants:
            print(f"\n--- Could not find any data for {pollutant.upper()} ---")
            continue

        print(f"\n--- Analyzing {pollutant.upper()} ---")
        if not combined_df.empty and pollutant in combined_df.columns:
            historical_df = combined_df[combined_df['datetime'] <= now_utc].copy()
            api_forecast_df = combined_df[combined_df['datetime'] > now_utc].copy()
            sarima_forecast_df = forecast_pollutant_sarima(historical_df, pollutant, hours_ahead=24)
            plot_pollutant(historical_df, api_forecast_df, sarima_forecast_df, pollutant, city_name)
            if sarima_forecast_df is not None:
                threshold = HEALTH_THRESHOLDS.get(pollutant, 150)
                generate_alerts(sarima_forecast_df, pollutant, threshold)
        else:
            print(f"⚠️ Historical data for {pollutant.upper()} not available. Prediction is not possible.")
            current_data = real_time_aq_data.get(pollutant)
            if current_data and 'v' in current_data:
                plot_realtime_only(pollutant, current_data, now_utc, city_name)
            else:
                print(f"Could not find a current value for {pollutant.upper()} to display.")
else:
    print("Could not fetch any real-time data to begin analysis.")