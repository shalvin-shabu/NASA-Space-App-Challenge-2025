from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime as dt
import numpy as np
import os
import warnings
import json
import random

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("🌍 Starting Air Quality Forecast Server...")

# No API credentials needed - we generate mock data
REQUIRED_POLLUTANTS = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

class AirQualitySimulator:
    def __init__(self):
        self.health_thresholds = {
            'pm25': 35,   # µg/m³
            'pm10': 50,   # µg/m³
            'o3': 70,     # µg/m³
            'no2': 40,    # µg/m³
            'so2': 20,    # µg/m³
            'co': 4000    # µg/m³
        }
    
    def generate_city_profile(self, city_name):
        """Generate city-specific air quality profiles"""
        profiles = {
            'New Delhi': {
                'pm25_base': 80, 'pm10_base': 120, 'o3_base': 45,
                'no2_base': 60, 'so2_base': 15, 'co_base': 2000,
                'pollution_level': 'high'
            },
            'Los Angeles': {
                'pm25_base': 25, 'pm10_base': 40, 'o3_base': 50,
                'no2_base': 30, 'so2_base': 8, 'co_base': 1200,
                'pollution_level': 'moderate'
            },
            'Canberra': {
                'pm25_base': 12, 'pm10_base': 20, 'o3_base': 25,
                'no2_base': 15, 'so2_base': 5, 'co_base': 800,
                'pollution_level': 'low'
            },
            'London': {
                'pm25_base': 18, 'pm10_base': 25, 'o3_base': 35,
                'no2_base': 35, 'so2_base': 6, 'co_base': 900,
                'pollution_level': 'moderate'
            },
            'Tokyo': {
                'pm25_base': 22, 'pm10_base': 35, 'o3_base': 40,
                'no2_base': 25, 'so2_base': 7, 'co_base': 1100,
                'pollution_level': 'moderate'
            }
        }
        return profiles.get(city_name, {
            'pm25_base': 20, 'pm10_base': 30, 'o3_base': 35,
            'no2_base': 25, 'so2_base': 8, 'co_base': 1000,
            'pollution_level': 'moderate'
        })
    
    def generate_mock_data(self, location_name, days=7):
        """Generate realistic mock air quality data"""
        print(f"📊 Generating mock data for {location_name}...")
        
        city_profile = self.generate_city_profile(location_name)
        end_time = dt.datetime.utcnow()
        start_time = end_time - dt.timedelta(days=days)
        
        # Generate hourly data points
        dates = pd.date_range(start=start_time, end=end_time + dt.timedelta(hours=24), freq='H')
        data = []
        
        for i, timestamp in enumerate(dates):
            record = {'datetime': timestamp}
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Time-based patterns
            rush_hour_effect = 1.0
            if (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19):  # Rush hours
                rush_hour_effect = 1.3
            elif 0 <= hour_of_day <= 5:  # Night time
                rush_hour_effect = 0.7
            
            # Weekend effect
            weekend_effect = 0.9 if day_of_week >= 5 else 1.0
            
            # Seasonal variation (simulated)
            seasonal_variation = 1.0 + 0.2 * np.sin(i / 24 * 2 * np.pi / 30)  # Monthly cycle
            
            for pollutant in REQUIRED_POLLUTANTS:
                base_value = city_profile[f'{pollutant}_base']
                
                # Add realistic variations
                diurnal_variation = 1.0 + 0.3 * np.sin((hour_of_day - 6) * 2 * np.pi / 24)
                random_noise = random.uniform(-0.1, 0.1) * base_value
                
                # Calculate final value
                final_value = (base_value * diurnal_variation * rush_hour_effect * 
                             weekend_effect * seasonal_variation + random_noise)
                
                # Ensure non-negative values
                record[pollutant] = max(0, final_value)
            
            data.append(record)
        
        print(f"✅ Generated {len(data)} data points for {location_name}")
        return data
    
    def analyze_air_quality(self, data, location_name):
        """Analyze air quality data and generate insights"""
        print(f"🔍 Analyzing air quality for {location_name}...")
        
        df = pd.DataFrame(data)
        analysis_results = []
        alerts = []
        current_values = {}
        
        # Calculate statistics for each pollutant
        for pollutant in REQUIRED_POLLUTANTS:
            if pollutant in df.columns:
                values = df[pollutant].dropna()
                if len(values) > 0:
                    current_val = values.iloc[-1]
                    avg_val = values.mean()
                    max_val = values.max()
                    min_val = values.min()
                    
                    current_values[pollutant] = current_val
                    
                    # Create analysis result
                    result = f"📊 {pollutant.upper()}: Current {current_val:.1f} µg/m³, " \
                           f"Avg {avg_val:.1f} µg/m³, Range {min_val:.1f}-{max_val:.1f} µg/m³"
                    analysis_results.append(result)
                    
                    # Check for health alerts
                    threshold = self.health_thresholds.get(pollutant, 50)
                    if current_val > threshold:
                        alert_msg = f"🚨 High {pollutant.upper()} levels: {current_val:.1f} µg/m³ " \
                                  f"(Threshold: {threshold} µg/m³)"
                        alerts.append(alert_msg)
        
        # Overall assessment
        if 'pm25' in current_values:
            pm25_level = current_values['pm25']
            if pm25_level > 75:
                assessment = "🎯 CRITICAL: Very poor air quality. Avoid outdoor activities."
                alerts.append("🚨 CRITICAL PM2.5 levels - Health advisory issued")
            elif pm25_level > 35:
                assessment = "🎯 POOR: Air quality is unhealthy. Limit outdoor exposure."
            elif pm25_level > 25:
                assessment = "🎯 MODERATE: Air quality is acceptable."
            else:
                assessment = "🎯 GOOD: Air quality is satisfactory."
        else:
            assessment = "🎯 Air quality analysis completed."
        
        analysis_results.append(assessment)
        
        # Add trend analysis
        trends = self.analyze_trends(df)
        analysis_results.extend(trends)
        
        print(f"✅ Analysis completed for {location_name}")
        return analysis_results, alerts, current_values
    
    def analyze_trends(self, df):
        """Analyze trends in the air quality data"""
        trends = []
        
        for pollutant in REQUIRED_POLLUTANTS:
            if pollutant in df.columns and len(df) >= 24:
                # Compare last 6 hours vs previous 6 hours
                recent_data = df[pollutant].tail(6)
                previous_data = df[pollutant].tail(12).head(6)
                
                if len(recent_data) >= 3 and len(previous_data) >= 3:
                    recent_avg = recent_data.mean()
                    previous_avg = previous_data.mean()
                    
                    if recent_avg > previous_avg * 1.1:
                        trends.append(f"📈 {pollutant.upper()} trend: Increasing")
                    elif recent_avg < previous_avg * 0.9:
                        trends.append(f"📉 {pollutant.upper()} trend: Improving")
                    else:
                        trends.append(f"➡️ {pollutant.upper()} trend: Stable")
        
        return trends
    
    def generate_forecast(self, data, hours=24):
        """Generate simple forecast based on recent trends"""
        df = pd.DataFrame(data)
        forecast_data = []
        
        last_point = df.iloc[-1].copy()
        base_time = last_point['datetime']
        
        for i in range(1, hours + 1):
            forecast_point = {'datetime': base_time + dt.timedelta(hours=i)}
            
            for pollutant in REQUIRED_POLLUTANTS:
                if pollutant in df.columns:
                    # Simple forecast: slight random walk from current value
                    current_val = last_point[pollutant]
                    trend = random.uniform(-0.05, 0.05)  # Small random trend
                    forecast_point[pollutant] = max(0, current_val * (1 + trend))
            
            forecast_data.append(forecast_point)
        
        return forecast_data

# Initialize the simulator
simulator = AirQualitySimulator()

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Air Quality Backend</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Air Quality Forecast Backend</h1>
            <div class="status">
                <p><strong>Status:</strong> ✅ Backend server is running successfully</p>
                <p><strong>Port:</strong> 5000</p>
                <p><strong>Mode:</strong> Mock Data Simulation (No APIs Required)</p>
            </div>
            <div class="endpoint">
                <p><strong>API Endpoint:</strong> POST http://127.0.0.1:5000/api/location</p>
                <p><strong>Health Check:</strong> GET http://127.0.0.1:5000/api/health</p>
                <p><strong>Cities Available:</strong> New Delhi, Los Angeles, Canberra, London, Tokyo</p>
            </div>
            <p>This backend generates realistic mock air quality data for demonstration purposes.</p>
        </div>
    </body>
    </html>
    """

@app.route('/api/location', methods=['POST', 'OPTIONS'])
def analyze_location():
    print("📍 /api/location endpoint called")
    
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        print(f"📨 Received data: {data}")
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        if 'lat' not in data or 'lng' not in data:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        lat_input = float(data['lat'])
        lon_input = float(data['lng'])
        location_name = data.get('name', 'Unknown Location')
        
        print(f"🎯 Starting analysis for: {location_name} ({lat_input}, {lon_input})")
        
        # Step 1: Generate mock data
        historical_data = simulator.generate_mock_data(location_name, days=7)
        
        # Step 2: Generate forecast
        forecast_data = simulator.generate_forecast(historical_data, hours=24)
        
        # Combine historical and forecast data
        all_data = historical_data + forecast_data
        
        # Step 3: Analyze the data
        analysis_results, alerts, current_values = simulator.analyze_air_quality(all_data, location_name)
        
        # Prepare response
        response_data = {
            'message': f'Air quality analysis completed for {location_name}',
            'location': location_name,
            'latitude': lat_input,
            'longitude': lon_input,
            'results': analysis_results,
            'alerts': alerts,
            'current_values': current_values,
            'data_points': len(all_data),
            'status': 'success',
            'timestamp': dt.datetime.utcnow().isoformat(),
            'features': {
                'historical_days': 7,
                'forecast_hours': 24,
                'pollutants_analyzed': REQUIRED_POLLUTANTS,
                'data_source': 'mock_simulation'
            }
        }
        
        print(f"✅ Analysis completed successfully for {location_name}")
        print(f"📊 Generated {len(all_data)} data points")
        print(f"🚨 Found {len(alerts)} alerts")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'Air Quality Forecast API',
        'mode': 'mock_data_simulation',
        'timestamp': dt.datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'features': {
            'pollutants': REQUIRED_POLLUTANTS,
            'cities': ['New Delhi', 'Los Angeles', 'Canberra', 'London', 'Tokyo'],
            'data_points': '7 days historical + 24 hours forecast'
        }
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the server is working"""
    return jsonify({
        'message': '✅ Air Quality Backend is working!',
        'endpoints': {
            'GET /': 'Home page',
            'POST /api/location': 'Analyze air quality for location',
            'GET /api/health': 'Health check',
            'GET /api/test': 'This test endpoint'
        },
        'example_request': {
            'name': 'New Delhi',
            'lat': 28.6139,
            'lng': 77.2090
        }
    })

@app.route('/api/cities', methods=['GET'])
def available_cities():
    """List available cities with their pollution profiles"""
    cities = {
        'New Delhi': {'pollution_level': 'high', 'description': 'High particulate matter levels'},
        'Los Angeles': {'pollution_level': 'moderate', 'description': 'Moderate pollution with ozone concerns'},
        'Canberra': {'pollution_level': 'low', 'description': 'Generally good air quality'},
        'London': {'pollution_level': 'moderate', 'description': 'Moderate pollution, NO2 concerns'},
        'Tokyo': {'pollution_level': 'moderate', 'description': 'Moderate pollution levels'}
    }
    return jsonify({'cities': cities})

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    print("📍 Available endpoints:")
    print("   - GET  /              : Home page")
    print("   - POST /api/location   : Analyze location (main endpoint)")
    print("   - GET  /api/health     : Health check") 
    print("   - GET  /api/test       : Test endpoint")
    print("   - GET  /api/cities     : List available cities")
    print("\n🎯 Supported cities: New Delhi, Los Angeles, Canberra, London, Tokyo")
    print("📊 Mode: Mock Data Simulation (No external APIs required)")
    print("💡 Use the HTML frontend or send POST requests to /api/location")
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=True)