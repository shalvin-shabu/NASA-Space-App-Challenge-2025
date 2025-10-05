from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import datetime as dt
import numpy as np
import warnings
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==============================
# 🔐 API Credentials & Config
# ==============================
METEO_USER = "menon_prithvishankar"
METEO_PASS = "Q6bRXp5GWf1mLoLa7Jz1"
MODEL_FILENAME = "rainfall_forecast_model.joblib"

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
        response = requests.get(url, headers=headers)
        response.raise_for_status()
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
        response = requests.get(url, auth=(username, password))
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'OK' or not data.get('data'):
            print(f"❌ Meteomatics API Error: {data.get('message')}")
            return pd.DataFrame()

        all_params_data = {}
        for param_data in data['data']:
            param_name = param_data['parameter']
            values = [d['value'] for d in param_data['coordinates'][0]['dates']]
            all_params_data[param_name] = values

        df = pd.DataFrame(all_params_data)
        df['datetime'] = [d['date'] for d in data['data'][0]['coordinates'][0]['dates']]
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.rename(columns={
            't_2m:C': 'temperature', 
            'relative_humidity_2m:p': 'humidity', 
            'precip_1h:mm': RAINFALL_COL
        }, inplace=True)

        print(f"✅ Fetched data. Total points: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return pd.DataFrame()

def train_and_save_model(df, variable_name, model_path):
    """Trains a SARIMA model on the full dataset and saves it to a file."""
    print(f"Training final model and saving to '{model_path}'...")
    time_series = df.set_index('datetime')[variable_name].asfreq('H').dropna()
    if len(time_series) < 48:
        print(f"❌ Not enough historical data to train the final model.")
        return None
    try:
        model = ARIMA(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        model_fit = model.fit()
        joblib.dump(model_fit, model_path)
        print(f"✅ Model successfully trained and saved.")
        return model_fit
    except Exception as e:
        print(f"❌ Final model training/saving failed: {e}")
        return None

def load_and_forecast(model_path, last_timestamp, hours_ahead=24):
    """Loads a saved SARIMA model from a file and generates a forecast."""
    print(f"Loading model from '{model_path}' to generate forecast...")
    try:
        model_fit = joblib.load(model_path)
        forecast = model_fit.get_forecast(steps=hours_ahead)
        forecast_df = forecast.summary_frame()
        forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
        future_time = [last_timestamp + dt.timedelta(hours=i + 1) for i in range(hours_ahead)]
        return pd.DataFrame({
            "datetime": future_time,
            f"{RAINFALL_COL}_forecast": forecast_df['mean'].values
        })
    except FileNotFoundError:
        print(f"❌ Model file not found at '{model_path}'. Please train the model first.")
        return None
    except Exception as e:
        print(f"❌ Model loading or forecasting failed: {e}")
        return None

def generate_rain_alert(forecast_df):
    """Generates an alert if moderate or heavy rain is predicted."""
    if forecast_df is None:
        return "No forecast data available"
    
    forecast_col = f"{RAINFALL_COL}_forecast"
    moderate_rain = forecast_df[forecast_df[forecast_col] >= RAIN_THRESHOLDS['moderate']]
    heavy_rain = forecast_df[forecast_df[forecast_col] >= RAIN_THRESHOLDS['heavy']]
    
    if not heavy_rain.empty:
        return "🚨 ALERT: Chance of 'very wet' conditions! Heavy rain predicted."
    elif not moderate_rain.empty:
        return "⚠ ALERT: Chance of moderate rain predicted."
    else:
        return "✅ FORECAST: Conditions look dry for the next 24 hours."

def describe_current_climate(latest_reading):
    """Analyzes the latest data point to provide a human-readable climate description."""
    if latest_reading is None or latest_reading.empty:
        return "Could not determine current conditions."

    temp = latest_reading['temperature'].iloc[0]
    humidity = latest_reading['humidity'].iloc[0]
    rain = latest_reading[RAINFALL_COL].iloc[0]

    if temp > 35: 
        temp_desc = "Very Hot"
    elif temp > 28: 
        temp_desc = "Hot"
    elif temp > 20: 
        temp_desc = "Warm"
    elif temp > 10: 
        temp_desc = "Cool"
    else: 
        temp_desc = "Cold"

    if humidity > 75: 
        humidity_desc = "Humid"
    elif humidity < 40: 
        humidity_desc = "Dry"
    else: 
        humidity_desc = ""

    climate_summary = f"{temp_desc} and {humidity_desc}" if humidity_desc else temp_desc
    if rain > 0.5: 
        climate_summary += " with ongoing rain"

    return f"Current Condition: {climate_summary} ({temp:.1f}°C, {humidity:.0f}% humidity)"

# ==============================
# Flask Routes
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_location():
    try:
        data = request.get_json()
        lat = float(data['lat'])
        lon = float(data['lon'])
        location_name = data.get('location_name', 'Unknown Location')
        
        print(f"Analyzing location: {location_name} ({lat}, {lon})")
        
        # Get weather data
        combined_df = get_weather_data(lat, lon, WEATHER_VARIABLES, METEO_USER, METEO_PASS)
        
        if combined_df.empty:
            return jsonify({
                'success': False,
                'error': 'Could not fetch weather data from API'
            })
        
        # Process data
        now_utc = pd.Timestamp.utcnow()
        historical_df = combined_df[combined_df['datetime'] <= now_utc].copy()
        api_forecast_df = combined_df[combined_df['datetime'] > now_utc].copy()
        
        # Train model
        train_and_save_model(historical_df, RAINFALL_COL, MODEL_FILENAME)
        
        # Generate forecast
        last_known_timestamp = historical_df['datetime'].iloc[-1]
        our_forecast_df = load_and_forecast(MODEL_FILENAME, last_known_timestamp, hours_ahead=24)
        
        # Prepare response data
        response_data = {
            'success': True,
            'location_name': location_name,
            'coordinates': f"Latitude: {lat}, Longitude: {lon}",
            'current_condition': describe_current_climate(historical_df.tail(1)),
            'alert': generate_rain_alert(our_forecast_df) if our_forecast_df is not None else "No forecast available",
            'historical_data': historical_df.tail(24).to_dict('records'),
            'api_forecast': api_forecast_df.to_dict('records'),
            'our_forecast': our_forecast_df.to_dict('records') if our_forecast_df is not None else []
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)