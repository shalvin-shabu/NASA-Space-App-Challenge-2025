# ============================================
# 🌦 Climate & Rainfall Forecast App
# With Model Export/Import using Joblib
# Model is trained on 1 month (30 days) of historical data.
# ============================================

# Ensure necessary libraries are installed
# !pip install requests pandas plotly scikit-learn statsmodels joblib

import requests
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import joblib # Import joblib for saving and loading the model

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# ==============================
# 🔐 API Credentials & Config
# ==============================
METEO_USER = "menon_prithvishankar"  # Replace with your Meteomatics username
METEO_PASS = "Q6bRXp5GWf1mLoLa7Jz1"  # Replace with your Meteomatics password
MODEL_FILENAME = "rainfall_forecast_model.joblib" # Filename for the saved model

WEATHER_VARIABLES = ['t_2m:C', 'relative_humidity_2m:p', 'precip_1h:mm']
RAINFALL_COL = 'rainfall'
RAIN_THRESHOLDS = {'moderate': 2.5, 'heavy': 7.6}

# ==============================
# ⚙ Helper Functions
# ==============================

def get_location_name(lat, lon):
    """Uses OpenStreetMap's Nominatim to get a city name from coordinates."""
    headers = {'User-Agent': 'RainfallForecastApp/1.0'}
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    try:
        response = requests.get(url, headers=headers); response.raise_for_status()
        data = response.json()
        address = data.get('address', {})
        location = address.get('city', address.get('town', address.get('village', 'Unknown Location')))
        return location
    except Exception:
        return "Unknown Location"

def get_weather_data(lat, lon, variables, username, password):
    """Fetches historical (30 days) and forecast (24 hours) data."""
    params_str = ",".join(variables)
    end_time = dt.datetime.utcnow() + dt.timedelta(hours=24)
    start_time = dt.datetime.utcnow() - dt.timedelta(days=30)
    time_str = f"{start_time.isoformat(timespec='seconds')}Z--{end_time.isoformat(timespec='seconds')}Z:PT1H"
    location_str = f"{lat},{lon}"
    url = f"https://api.meteomatics.com/{time_str}/{params_str}/{location_str}/json"

    print("Querying Meteomatics API for weather data (last 30 days)...")
    try:
        response = requests.get(url, auth=(username, password)); response.raise_for_status()
        data = response.json()
        if data.get('status') != 'OK' or not data.get('data'):
            print(f"❌ Meteomatics API Error: {data.get('message')}"); return pd.DataFrame()

        all_params_data = {}
        for param_data in data['data']:
            param_name = param_data['parameter']
            values = [d['value'] for d in param_data['coordinates'][0]['dates']]
            all_params_data[param_name] = values

        df = pd.DataFrame(all_params_data)
        df['datetime'] = [d['date'] for d in data['data'][0]['coordinates'][0]['dates']]
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.rename(columns={'t_2m:C': 'temperature', 'relative_humidity_2m:p': 'humidity', 'precip_1h:mm': RAINFALL_COL}, inplace=True)

        print(f"✅ Fetched data. Total points: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ An error occurred: {e}"); return pd.DataFrame()

def describe_current_climate(latest_reading):
    """Analyzes the latest data point to provide a human-readable climate description."""
    if latest_reading is None or latest_reading.empty:
        return "Could not determine current conditions."

    temp, humidity, rain = latest_reading['temperature'].iloc[0], latest_reading['humidity'].iloc[0], latest_reading[RAINFALL_COL].iloc[0]

    if temp > 35: temp_desc = "Very Hot"
    elif temp > 28: temp_desc = "Hot"
    elif temp > 20: temp_desc = "Warm"
    elif temp > 10: temp_desc = "Cool"
    else: temp_desc = "Cold"

    if humidity > 75: humidity_desc = "Humid"
    elif humidity < 40: humidity_desc = "Dry"
    else: humidity_desc = ""

    climate_summary = f"{temp_desc} and {humidity_desc}" if humidity_desc else temp_desc
    if rain > 0.5: climate_summary += " with ongoing rain"

    return f"Current Condition: {climate_summary} ({temp:.1f}°C, {humidity:.0f}% humidity)"

def forecast_sarima_for_eval(df, variable_name, hours_ahead=24):
    """Internal function for evaluation only. Does not save the model."""
    time_series = df.set_index('datetime')[variable_name].asfreq('H').dropna()
    if len(time_series) < 48:
        print(f"Not enough data to forecast {variable_name}."); return None
    try:
        model = ARIMA(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=hours_ahead)
        forecast_df = forecast.summary_frame()
        forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
        future_time = [time_series.index[-1] + dt.timedelta(hours=i + 1) for i in range(hours_ahead)]
        return pd.DataFrame({"datetime": future_time, f"{variable_name}_forecast": forecast_df['mean'].values})
    except Exception as e:
        print(f"❌ SARIMA model failed during evaluation: {e}"); return None

def evaluate_model(historical_data, variable_name):
    """Splits data, trains a temporary model, and evaluates its performance."""
    historical_data = historical_data.dropna(subset=[variable_name])
    if len(historical_data) < 72:
        print("Not enough data to perform an evaluation."); return
    train_df, test_df = historical_data.iloc[:-24], historical_data.iloc[-24:]
    print("Evaluating model on the last 24 hours of known data...")
    validation_forecast_df = forecast_sarima_for_eval(train_df, variable_name, hours_ahead=24)
    if validation_forecast_df is not None:
        mae = mean_absolute_error(test_df[variable_name], validation_forecast_df[f"{variable_name}_forecast"])
        rmse = np.sqrt(mean_squared_error(test_df[variable_name], validation_forecast_df[f"{variable_name}_forecast"]))
        print("✅ Evaluation Complete:")
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f} mm/h")
        print(f"  - Mean Absolute Error (MAE):      {mae:.4f} mm/h")
    else:
        print("❌ Model evaluation failed.")


def train_and_save_model(df, variable_name, model_path):
    """Trains a SARIMA model on the full dataset and saves it to a file."""
    print(f"Training final model and saving to '{model_path}'...")
    time_series = df.set_index('datetime')[variable_name].asfreq('H').dropna()
    if len(time_series) < 48:
        print(f"❌ Not enough historical data to train the final model."); return None
    try:
        model = ARIMA(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        model_fit = model.fit()
        joblib.dump(model_fit, model_path)
        print(f"✅ Model successfully trained and saved.")
        return model_fit
    except Exception as e:
        print(f"❌ Final model training/saving failed: {e}"); return None

def load_and_forecast(model_path, last_timestamp, hours_ahead=24):
    """Loads a saved SARIMA model from a file and generates a forecast."""
    print(f"Loading model from '{model_path}' to generate forecast...")
    try:
        model_fit = joblib.load(model_path)
        forecast = model_fit.get_forecast(steps=hours_ahead)
        forecast_df = forecast.summary_frame()
        forecast_df['mean'] = forecast_df['mean'].clip(lower=0) # Ensure no negative rainfall
        future_time = [last_timestamp + dt.timedelta(hours=i + 1) for i in range(hours_ahead)]
        return pd.DataFrame({
            "datetime": future_time,
            f"{RAINFALL_COL}_forecast": forecast_df['mean'].values
        })
    except FileNotFoundError:
        print(f"❌ Model file not found at '{model_path}'. Please train the model first.")
        return None
    except Exception as e:
        print(f"❌ Model loading or forecasting failed: {e}"); return None

def plot_forecast(historical_df, api_forecast_df, our_forecast_df, city):
    """Plots the rainfall data and forecasts using a compatible renderer."""
    title = f"Rainfall 24-Hour Forecast – {city}"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=historical_df["datetime"], y=historical_df[RAINFALL_COL], name='Historical Data', marker_color='blue'))
    fig.add_trace(go.Bar(x=api_forecast_df["datetime"], y=api_forecast_df[RAINFALL_COL], name='API Forecast', marker_color='#87CEEB'))
    if our_forecast_df is not None:
        fig.add_trace(go.Scatter(x=our_forecast_df["datetime"], y=our_forecast_df[f"{RAINFALL_COL}_forecast"], mode='lines+markers', name='Our Model Forecast', line=dict(color='red', dash='dot')))
        zoom_start = historical_df["datetime"].iloc[-1] - dt.timedelta(hours=6)
        zoom_end = our_forecast_df["datetime"].iloc[-1] + dt.timedelta(hours=1)
        full_view_start, full_view_end = historical_df["datetime"].min(), zoom_end
    else:
        full_view_start = historical_df["datetime"].min()
        full_view_end = api_forecast_df["datetime"].max() if not api_forecast_df.empty else historical_df["datetime"].max()
        zoom_start, zoom_end = (historical_df["datetime"].iloc[-1] - dt.timedelta(hours=6)), full_view_end
    
    fig.update_layout(
        title=title, 
        xaxis_title="Time (UTC)", 
        yaxis_title="Rainfall (mm/h)", 
        legend_title_text='Data Source',
        height=600,
        updatemenus=[dict(
            type="buttons", 
            direction="right", 
            x=0.57, 
            xanchor="left", 
            y=1.15, 
            yanchor="top", 
            buttons=list([
                dict(label="Full View", method="relayout", args=[{"xaxis.range": [full_view_start, full_view_end]}]),
                dict(label="Zoom to Forecast", method="relayout", args=[{"xaxis.range": [zoom_start, zoom_end]}])
            ])
        )]
    )
    
    # Use a more compatible approach for displaying the plot
    try:
        # Try using the default renderer
        fig.show()
    except Exception as e:
        print(f"Note: Could not display interactive plot: {e}")
        print("The analysis completed successfully. Consider saving the plot as HTML or image.")
        # Alternative: Save as HTML file
        fig.write_html("rainfall_forecast_plot.html")
        print("Plot saved as 'rainfall_forecast_plot.html'")

def generate_rain_alert(df):
    """Generates an alert if moderate or heavy rain is predicted."""
    if df is None:
        return
    forecast_col = f"{RAINFALL_COL}_forecast"
    moderate_rain = df[df[forecast_col] >= RAIN_THRESHOLDS['moderate']]
    heavy_rain = df[df[forecast_col] >= RAIN_THRESHOLDS['heavy']]
    if not heavy_rain.empty: 
        print(f"\n🚨 ALERT: Chance of 'very wet' conditions! Heavy rain predicted.")
    elif not moderate_rain.empty: 
        print(f"\n⚠ ALERT: Chance of moderate rain predicted.")
    else: 
        print(f"\n✅ FORECAST: Conditions look dry for the next 24 hours.")

# ==============================
# ▶ Main Execution Block
# ==============================
if __name__ == "__main__":
    while True:
        try:
            lat_input = float(input("Enter the latitude (e.g., 9.9312 for Kochi): "))
            lon_input = float(input("Enter the longitude (e.g., 76.2673 for Kochi): "))
            if -90 <= lat_input <= 90 and -180 <= lon_input <= 180: 
                break
            else: 
                print("❌ Invalid range.")
        except ValueError:
            print("❌ Invalid input.")

    city_name = get_location_name(lat_input, lon_input)
    print(f"\n▶ Running analysis for {city_name} ({lat_input}, {lon_input})...")

    combined_df = get_weather_data(lat_input, lon_input, WEATHER_VARIABLES, METEO_USER, METEO_PASS)

    if not combined_df.empty:
        now_utc = pd.Timestamp.utcnow()
        historical_df = combined_df[combined_df['datetime'] <= now_utc].copy()
        api_forecast_df = combined_df[combined_df['datetime'] > now_utc].copy()

        # --- Step 1: Evaluate model performance on past data ---
        print(f"\n{'-'*40}\nModel Performance Evaluation\n{'-'*40}")
        evaluate_model(historical_df, RAINFALL_COL)
        print(f"{'-'*40}\n")

        # --- Step 2: Describe current conditions ---
        latest_reading = historical_df.tail(1)
        climate_description = describe_current_climate(latest_reading)
        print(f"Current Climate Analysis\n{'-'*40}\n{climate_description}\n{'-'*40}\n")

        # --- Step 3: Train the final model on all historical data and SAVE it ---
        train_and_save_model(historical_df, RAINFALL_COL, MODEL_FILENAME)
        
        # --- Step 4: LOAD the saved model and generate the future forecast ---
        last_known_timestamp = historical_df['datetime'].iloc[-1]
        our_sarima_forecast_df = load_and_forecast(MODEL_FILENAME, last_known_timestamp, hours_ahead=24)

        # --- Step 5: Visualize results and generate alerts ---
        plot_forecast(historical_df, api_forecast_df, our_sarima_forecast_df, city_name)

        if our_sarima_forecast_df is not None:
            generate_rain_alert(our_sarima_forecast_df)
    else:
        print("\n❌ Could not fetch weather data.")