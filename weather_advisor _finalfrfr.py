import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyinputplus as pyip
import re
from datetime import datetime, timedelta
import json
import os
import platform
import time
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# API configuration
API_KEY = "a2d2296beda67fc9855b4eb3559e24b3"
 
BASE_URL = "https://api.openweathermap.org/data/2.5"

def get_weather_data(location, forecast_days=5):
    """
    Retrieve weather data for a specified location.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-5)
        
    Returns:
        dict: Weather data including current conditions and forecast
    """
    # Ensure forecast days is within acceptable range
    forecast_days = max(1, min(forecast_days, 5))
    
    try:
        # Get current weather
        current_url = f"{BASE_URL}/weather?q={location}&units=metric&appid={API_KEY}"
        current_response = requests.get(current_url)
        
        if current_response.status_code != 200:
            if current_response.status_code == 404:
                return {"error": "Location not found. Please check the spelling and try again."}
            else:
                return {"error": f"API Error: {current_response.status_code}"}
        
        current_data = current_response.json()
        
        # Get forecast data
        forecast_url = f"{BASE_URL}/forecast?q={location}&units=metric&appid={API_KEY}"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code != 200:
            return {"error": f"Forecast API Error: {forecast_response.status_code}"}
        
        forecast_data = forecast_response.json()
        
        # Process and structure the data
        processed_data = {
            "location": {
                "name": current_data["name"],
                "country": current_data["sys"]["country"],
                "coordinates": {
                    "lat": current_data["coord"]["lat"],
                    "lon": current_data["coord"]["lon"]
                }
            },
            "current": {
                "timestamp": datetime.fromtimestamp(current_data["dt"]).strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": current_data["main"]["temp"],
                "feels_like": current_data["main"]["feels_like"],
                "humidity": current_data["main"]["humidity"],
                "pressure": current_data["main"]["pressure"],
                "wind": {
                    "speed": current_data["wind"]["speed"],
                    "direction": current_data["wind"].get("deg", 0)
                },
                "description": current_data["weather"][0]["description"],
                "main": current_data["weather"][0]["main"],
                "icon": current_data["weather"][0]["icon"],
                "clouds": current_data.get("clouds", {}).get("all", 0),
                "rain": current_data.get("rain", {}).get("1h", 0),
                "visibility": current_data.get("visibility", 0) / 1000  # Convert to km
            },
            "forecast": []
        }
        
        # Group forecast by day
        daily_forecasts = {}
        today = datetime.now().date()
        
        for item in forecast_data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            day = dt.date()
            
            if (day - today).days >= forecast_days:
                continue
                
            if day not in daily_forecasts:
                daily_forecasts[day] = []
                
            daily_forecasts[day].append({
                "timestamp": dt.strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "description": item["weather"][0]["description"],
                "main": item["weather"][0]["main"],
                "icon": item["weather"][0]["icon"],
                "clouds": item["clouds"]["all"],
                "wind": {
                    "speed": item["wind"]["speed"],
                    "direction": item["wind"].get("deg", 0)
                },
                "pop": item.get("pop", 0) * 100,  # Probability of precipitation as percentage
                "rain": item.get("rain", {}).get("3h", 0),
                "snow": item.get("snow", {}).get("3h", 0),
                "hour": dt.hour
            })
        
        # Calculate daily stats and add to processed data
        for day, forecasts in sorted(daily_forecasts.items()):
            daily_temps = [f["temperature"] for f in forecasts]
            daily_humidity = [f["humidity"] for f in forecasts]
            daily_clouds = [f["clouds"] for f in forecasts]
            daily_pop = [f["pop"] for f in forecasts]
            
            # Check if we have rain data
            daily_rain = [f["rain"] for f in forecasts if "rain" in f]
            
            daily_summary = {
                "date": day.strftime('%Y-%m-%d'),
                "day_name": day.strftime('%A'),
                "temperature": {
                    "min": min(daily_temps),
                    "max": max(daily_temps),
                    "avg": sum(daily_temps) / len(daily_temps)
                },
                "humidity": {
                    "min": min(daily_humidity),
                    "max": max(daily_humidity),
                    "avg": sum(daily_humidity) / len(daily_humidity)
                },
                "clouds": {
                    "min": min(daily_clouds),
                    "max": max(daily_clouds),
                    "avg": sum(daily_clouds) / len(daily_clouds)
                },
                "precipitation_chance": max(daily_pop),
                "hourly": forecasts
            }
            
            if daily_rain:
                daily_summary["rain"] = {
                    "total": sum(daily_rain),
                    "max": max(daily_rain)
                }
                
            processed_data["forecast"].append(daily_summary)
            
        return processed_data
    except requests.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error processing weather data: {str(e)}"}

def parse_weather_question(question):
    """
    Parse a natural language weather question.
    
    Args:
        question (str): User's weather-related question
        
    Returns:
        dict: Extracted information including location, time period, and weather attribute
    """
    # Convert to lowercase for easier matching
    question = question.lower()
    
    # Initialize empty result dictionary
    result = {
        "location": None,
        "time_period": "current",  # Default to current if not specified
        "weather_attribute": "general",  # Default to general weather if not specified
        "original_question": question
    }
    
    # Extract location - attempt to find city names
    # This is a simple approach - a more robust solution would use NER
    # Look for common patterns like "in [location]", "for [location]", etc.
    location_patterns = [
        r"(?:in|at|for|of|about) ([\w\s]+?)(?:$|\?|\.|\s(?:today|tomorrow|this|next|on))",
        r"([\w\s]+?)(?:'s|\s+weather)",
        r"(?:^|\s)([\w\s]+?)(?:$|\?|\.|weather)"
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, question)
        if match:
            potential_location = match.group(1).strip()
            # Filter out common words that might be mistaken for locations
            common_words = ["weather", "the", "a", "an", "it", "there", "here", "what", "how", "when"]
            if potential_location and potential_location.lower() not in common_words:
                result["location"] = potential_location
                break
    
    # Extract time period
    time_periods = {
        "current": ["now", "current", "present", "today", "at the moment", "currently"],
        "tomorrow": ["tomorrow"],
        "this week": ["this week", "coming week", "next few days", "this coming week"],
        "weekend": ["weekend", "this weekend", "coming weekend"]
    }
    
    for period, keywords in time_periods.items():
        if any(keyword in question for keyword in keywords):
            result["time_period"] = period
            break
            
    # Extract specific day references (next Monday, Tuesday, etc.)
    day_match = re.search(r"(?:next|this|on) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", question)
    if day_match:
        result["time_period"] = "specific_day"
        result["specific_day"] = day_match.group(1)
        
    # Extract date references (July 15, 2023-07-15, etc.)
    # This would need more complex parsing in a production system
        
    # Extract weather attributes
    weather_attributes = {
        "temperature": ["temperature", "temp", "hot", "cold", "warm", "chilly", "degrees", "°c", "°f", "celsius", "fahrenheit"],
        "precipitation": ["rain", "snow", "precipit", "shower", "storm", "wet", "umbrella", "raining", "snowing", "precipitation"],
        "humidity": ["humid", "humidity", "moisture", "damp"],
        "wind": ["wind", "breeze", "gust", "windy", "breezy", "gusty"],
        "clouds": ["cloud", "cloudy", "overcast", "clear", "sunny", "sun"],
        "visibility": ["visibility", "fog", "foggy", "mist", "misty", "haze", "hazy"],
        "pressure": ["pressure", "atmospheric", "barometric"],
        "uv": ["uv", "ultraviolet", "sun protection", "sunburn", "sun screen"]
    }
    
    for attribute, keywords in weather_attributes.items():
        if any(keyword in question for keyword in keywords):
            result["weather_attribute"] = attribute
            break
    
    return result

def generate_weather_response(parsed_question, weather_data):
    """
    Generate a natural language response to a weather question.
    
    Args:
        parsed_question (dict): Parsed question data
        weather_data (dict): Weather data
        
    Returns:
        str: Natural language response
    """
    # Check if there was an error fetching weather data
    if "error" in weather_data:
        return f"Sorry, I couldn't get the weather information: {weather_data['error']}"
    
    location = weather_data["location"]["name"]
    country = weather_data["location"]["country"]
    full_location = f"{location}, {country}"
    
    attribute = parsed_question["weather_attribute"]
    time_period = parsed_question["time_period"]
    
    # Handle current weather
    if time_period == "current":
        current = weather_data["current"]
        
        if attribute == "general":
            return (f"Currently in {full_location}, it's {current['temperature']}°C with "
                   f"{current['description']}. The humidity is {current['humidity']}% and "
                   f"wind speed is {current['wind']['speed']} m/s.")
        
        elif attribute == "temperature":
            return (f"The current temperature in {full_location} is {current['temperature']}°C, "
                   f"and it feels like {current['feels_like']}°C.")
        
        elif attribute == "precipitation":
            rain_amount = current.get("rain", 0)
            if rain_amount > 0:
                return f"It's currently raining in {full_location} with {rain_amount} mm of rain in the last hour."
            else:
                return f"There's currently no precipitation in {full_location}."
        
        elif attribute == "humidity":
            return f"The current humidity in {full_location} is {current['humidity']}%."
        
        elif attribute == "wind":
            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
            wind_dir_index = round(current['wind']['direction'] / 45)
            wind_dir_str = directions[wind_dir_index]
            return (f"The wind in {full_location} is currently blowing at {current['wind']['speed']} m/s "
                   f"from the {wind_dir_str} direction.")
        
        elif attribute == "clouds":
            if current["clouds"] < 20:
                sky_condition = "clear"
            elif current["clouds"] < 60:
                sky_condition = "partly cloudy"
            else:
                sky_condition = "cloudy"
            return f"The sky in {full_location} is currently {sky_condition} with {current['clouds']}% cloud cover."
        
        elif attribute == "visibility":
            if current["visibility"] >= 10:
                visibility_desc = "excellent"
            elif current["visibility"] >= 5:
                visibility_desc = "good"
            elif current["visibility"] >= 2:
                visibility_desc = "moderate"
            else:
                visibility_desc = "poor"
            return f"Visibility in {full_location} is currently {visibility_desc} at {current['visibility']} km."
        
        elif attribute == "pressure":
            return f"The barometric pressure in {full_location} is currently {current['pressure']} hPa."
        
        else:
            return f"Currently in {full_location}, it's {current['temperature']}°C with {current['description']}."
    
    # Handle tomorrow's weather
    elif time_period == "tomorrow" and weather_data["forecast"]:
        tomorrow = None
        today = datetime.now().date()
        tomorrow_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for day in weather_data["forecast"]:
            if day["date"] == tomorrow_date:
                tomorrow = day
                break
        
        if not tomorrow:
            return f"I don't have forecast data for tomorrow in {full_location}."
        
        if attribute == "general":
            return (f"Tomorrow in {full_location}, expect temperatures between "
                   f"{tomorrow['temperature']['min']:.1f}°C and {tomorrow['temperature']['max']:.1f}°C. "
                   f"The chance of precipitation is {tomorrow['precipitation_chance']:.0f}%.")
        
        elif attribute == "temperature":
            return (f"Tomorrow's temperature in {full_location} will range from "
                   f"{tomorrow['temperature']['min']:.1f}°C to {tomorrow['temperature']['max']:.1f}°C.")
        
        elif attribute == "precipitation":
            if "rain" in tomorrow and tomorrow["rain"]["total"] > 0:
                return (f"For tomorrow in {full_location}, expect rainfall with a total of about "
                       f"{tomorrow['rain']['total']:.1f} mm and a {tomorrow['precipitation_chance']:.0f}% chance of precipitation.")
            else:
                return (f"For tomorrow in {full_location}, there's a {tomorrow['precipitation_chance']:.0f}% "
                       f"chance of precipitation.")
        
        elif attribute == "humidity":
            return (f"Tomorrow's humidity in {full_location} will range from {tomorrow['humidity']['min']}% "
                   f"to {tomorrow['humidity']['max']}%, with an average of {tomorrow['humidity']['avg']:.0f}%.")
        
        elif attribute == "clouds":
            return f"Tomorrow in {full_location}, expect about {tomorrow['clouds']['avg']:.0f}% cloud coverage."
        
        else:
            return (f"Tomorrow in {full_location}, expect temperatures between "
                   f"{tomorrow['temperature']['min']:.1f}°C and {tomorrow['temperature']['max']:.1f}°C.")
    
    # Handle this week's weather
    elif time_period == "this week" and weather_data["forecast"]:
        if attribute == "general":
            overview = f"Weather forecast for {full_location} for the next {len(weather_data['forecast'])} days:\n\n"
            
            for day in weather_data["forecast"]:
                overview += (f"{day['day_name']}: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C, "
                           f"precipitation chance: {day['precipitation_chance']:.0f}%\n")
            
            return overview
        
        elif attribute == "temperature":
            min_temps = [day["temperature"]["min"] for day in weather_data["forecast"]]
            max_temps = [day["temperature"]["max"] for day in weather_data["forecast"]]
            
            avg_min = sum(min_temps) / len(min_temps)
            avg_max = sum(max_temps) / len(max_temps)
            
            return (f"This week in {full_location}, temperatures will range from about "
                   f"{min(min_temps):.1f}°C to {max(max_temps):.1f}°C, with average daily ranges "
                   f"from {avg_min:.1f}°C to {avg_max:.1f}°C.")
        
        elif attribute == "precipitation":
            precip_days = sum(1 for day in weather_data["forecast"] if day["precipitation_chance"] > 30)
            highest_chance = max(day["precipitation_chance"] for day in weather_data["forecast"])
            highest_day = next(day["day_name"] for day in weather_data["forecast"] 
                              if day["precipitation_chance"] == highest_chance)
            
            return (f"This week in {full_location}, there are {precip_days} days with significant "
                   f"chance of precipitation. The highest chance is {highest_chance:.0f}% on {highest_day}.")
        
        else:
            overview = f"Weather forecast for {full_location} for the next {len(weather_data['forecast'])} days:\n\n"
            
            for day in weather_data["forecast"]:
                overview += (f"{day['day_name']}: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C\n")
            
            return overview
    
    # Handle weekend forecast
    elif time_period == "weekend" and weather_data["forecast"]:
        weekend_days = []
        for day in weather_data["forecast"]:
            if day["day_name"] in ["Saturday", "Sunday"]:
                weekend_days.append(day)
        
        if not weekend_days:
            return f"I don't have forecast data for the weekend in {full_location}."
        
        if attribute == "general":
            overview = f"Weekend forecast for {full_location}:\n\n"
            
            for day in weekend_days:
                overview += (f"{day['day_name']}: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C, "
                           f"precipitation chance: {day['precipitation_chance']:.0f}%\n")
            
            return overview
        
        elif attribute == "temperature":
            min_temps = [day["temperature"]["min"] for day in weekend_days]
            max_temps = [day["temperature"]["max"] for day in weekend_days]
            
            return (f"This weekend in {full_location}, temperatures will range from "
                   f"{min(min_temps):.1f}°C to {max(max_temps):.1f}°C.")
        
        elif attribute == "precipitation":
            precip_days = sum(1 for day in weekend_days if day["precipitation_chance"] > 30)
            if precip_days > 0:
                return f"This weekend in {full_location}, there {'is' if precip_days == 1 else 'are'} {precip_days} day(s) with significant chance of precipitation."
            else:
                return f"This weekend in {full_location} should be dry with low chance of precipitation."
        
        else:
            overview = f"Weekend forecast for {full_location}:\n\n"
            
            for day in weekend_days:
                overview += (f"{day['day_name']}: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C\n")
            
            return overview
    
    # Handle specific day requests
    elif time_period == "specific_day" and "specific_day" in parsed_question and weather_data["forecast"]:
        target_day = parsed_question["specific_day"].capitalize()
        specific_forecast = None
        
        for day in weather_data["forecast"]:
            if day["day_name"].lower() == target_day.lower():
                specific_forecast = day
                break
        
        if not specific_forecast:
            return f"I don't have forecast data for {target_day} in {full_location}."
        
        if attribute == "general":
            return (f"On {target_day} in {full_location}, expect temperatures between "
                   f"{specific_forecast['temperature']['min']:.1f}°C and {specific_forecast['temperature']['max']:.1f}°C "
                   f"with a {specific_forecast['precipitation_chance']:.0f}% chance of precipitation.")
        
        elif attribute == "temperature":
            return (f"On {target_day} in {full_location}, the temperature will range from "
                   f"{specific_forecast['temperature']['min']:.1f}°C to {specific_forecast['temperature']['max']:.1f}°C.")
        
        elif attribute == "precipitation":
            return (f"On {target_day} in {full_location}, there's a {specific_forecast['precipitation_chance']:.0f}% "
                   f"chance of precipitation.")
        
        else:
            return (f"On {target_day} in {full_location}, expect temperatures between "
                   f"{specific_forecast['temperature']['min']:.1f}°C and {specific_forecast['temperature']['max']:.1f}°C.")
    
    # Fallback response
    return (f"Based on current data for {full_location}, the temperature is {weather_data['current']['temperature']}°C "
           f"with {weather_data['current']['description']}.")

def create_temperature_visualisation(weather_data, output_type='display'):
    """
    Create visualisation of temperature data.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
        
    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return None
    
    # Set up the figure and styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    location_name = f"{weather_data['location']['name']}, {weather_data['location']['country']}"
    
    # Current temperature reference
    current_temp = weather_data['current']['temperature']
    
    # Extract data for plotting
    dates = []
    temp_max = []
    temp_min = []
    temp_avg = []
    
    for day in weather_data['forecast']:
        dates.append(day['day_name'])
        temp_min.append(day['temperature']['min'])
        temp_max.append(day['temperature']['max'])
        temp_avg.append(day['temperature']['avg'])
    
    # Create x position for bars
    x = np.arange(len(dates))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, temp_min, width, label='Min Temp (°C)', color='lightblue', alpha=0.7)
    ax.bar(x, temp_avg, width, label='Avg Temp (°C)', color='skyblue', alpha=0.7)
    ax.bar(x + width, temp_max, width, label='Max Temp (°C)', color='darkblue', alpha=0.7)
    
    # Add a horizontal line for current temperature
    ax.axhline(y=current_temp, linestyle='--', color='red', alpha=0.7)
    ax.text(len(dates)-1, current_temp, f'Current: {current_temp:.1f}°C', 
            va='bottom', ha='right', color='red', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Temperature Forecast for {location_name}', fontsize=14, fontweight='bold')
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add overall range annotation
    overall_min = min(temp_min)
    overall_max = max(temp_max)
    ax.annotate(f'Temperature range: {overall_min:.1f}°C - {overall_max:.1f}°C',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=11, backgroundcolor='white', alpha=0.8)
    
    plt.tight_layout()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
        plt.close()

def create_precipitation_visualisation(weather_data, output_type='display'):
    """
    Create visualisation of precipitation data.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
        
    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return None
    
    # Set up the figure and styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    location_name = f"{weather_data['location']['name']}, {weather_data['location']['country']}"
    
    # Extract data for plotting
    dates = []
    precip_chance = []
    humidity_avg = []
    rain_amount = []
    
    for day in weather_data['forecast']:
        dates.append(day['day_name'])
        precip_chance.append(day['precipitation_chance'])
        humidity_avg.append(day['humidity']['avg'])
        
        # Some locations might not have rain data
        if 'rain' in day:
            rain_amount.append(day['rain']['total'])
        else:
            rain_amount.append(0)
    
    # Plot precipitation chance as bars
    x = np.arange(len(dates))
    bars = ax1.bar(x, precip_chance, color='skyblue', alpha=0.7, label='Precipitation Chance (%)')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(precip_chance):
        ax1.text(i, v + 2, f"{v:.0f}%", ha='center', fontsize=9)
    
    # Set primary axis labels
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Precipitation Chance (%)', fontsize=12)
    ax1.set_ylim(0, max(100, max(precip_chance) * 1.2))  # Cap at 100% with some headroom
    
    # Create secondary y-axis for rain amount
    ax2 = ax1.twinx()
    line = ax2.plot(x, rain_amount, 'o-', color='blue', linewidth=2, label='Rainfall (mm)')
    
    # Add rain amount annotations
    for i, v in enumerate(rain_amount):
        if v > 0:
            ax2.text(i, v + 0.5, f"{v:.1f}mm", ha='center', fontsize=9, color='blue')
    
    # Set secondary axis label
    ax2.set_ylabel('Expected Rainfall (mm)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set title and x-ticks
    plt.title(f'Precipitation Forecast for {location_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Add grid for better readability
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
        plt.close()
def create_wind_visualisation(weather_data, output_type='display'):
    """
    Create visualisation of wind data.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
        
    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return None
    
    # Set up the figure and styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    location_name = f"{weather_data['location']['name']}, {weather_data['location']['country']}"
    
    # Extract wind data from hourly forecasts
    hours = []
    wind_speeds = []
    wind_directions = []
    dates = []
    
    # First 24 hours forecast (8 data points, 3 hours apart)
    for day in weather_data['forecast'][:2]:  # First two days to get good coverage
        for hour_data in day['hourly'][:8]:  # First 8 readings (24 hours)
            dt = datetime.strptime(hour_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            hours.append(dt.strftime('%H:%M'))
            wind_speeds.append(hour_data['wind']['speed'])
            wind_directions.append(hour_data['wind']['direction'])
            dates.append(dt.strftime('%a'))
    
    # Plot wind speed as a line
    ax.plot(hours, wind_speeds, 'o-', color='teal', linewidth=2, label='Wind Speed (m/s)')
    
    # Add date labels at the bottom
    prev_date = None
    for i, date in enumerate(dates):
        if date != prev_date:
            ax.annotate(date, xy=(hours[i], -0.3), xycoords=('data', 'axes fraction'),
                       ha='center', va='top', fontsize=10)
            if prev_date is not None:
                ax.axvline(x=hours[i], color='gray', linestyle='--', alpha=0.3)
            prev_date = date
    
    # Add current wind speed reference
    current_wind = weather_data['current']['wind']['speed']
    ax.axhline(y=current_wind, linestyle='--', color='darkgreen', alpha=0.7)
    ax.text(hours[-1], current_wind, f'Current: {current_wind:.1f} m/s', 
            va='bottom', ha='right', color='darkgreen', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'Wind Speed Forecast for {location_name}', fontsize=14, fontweight='bold')
    
    # Add a legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
        plt.close()

def display_current_weather(weather_data):
    """
    Display current weather information in a formatted text output.
    
    Args:
        weather_data (dict): The processed weather data
    """
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    location = f"{weather_data['location']['name']}, {weather_data['location']['country']}"
    current = weather_data['current']
    
    print(f"\n{'='*50}")
    print(f"  CURRENT WEATHER FOR {location.upper()}")
    print(f"  {datetime.now().strftime('%A, %B %d, %Y at %H:%M')}")
    print(f"{'='*50}")
    print(f"Temperature: {current['temperature']:.1f}°C (Feels like: {current['feels_like']:.1f}°C)")
    print(f"Conditions: {current['description'].capitalize()}")
    print(f"Humidity: {current['humidity']}%")
    print(f"Pressure: {current['pressure']} hPa")
    print(f"Wind: {current['wind']['speed']} m/s")
    print(f"Visibility: {current['visibility']} km")
    print(f"Cloud Cover: {current['clouds']}%")
    
    # Add rain information if available
    if current.get('rain', 0) > 0:
        print(f"Rain in Last Hour: {current['rain']} mm")
    
    print(f"{'='*50}")

def display_forecast(weather_data):
    """
    Display weather forecast information in a formatted text output.
    
    Args:
        weather_data (dict): The processed weather data
    """
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    if not weather_data.get('forecast'):
        print("No forecast data available.")
        return
    
    location = f"{weather_data['location']['name']}, {weather_data['location']['country']}"
    
    print(f"\n{'='*60}")
    print(f"  WEATHER FORECAST FOR {location.upper()}")
    print(f"{'='*60}")
    
    for day in weather_data['forecast']:
        print(f"\n{day['day_name']} ({day['date']}):")
        print(f"  Temperature: {day['temperature']['min']:.1f}°C to {day['temperature']['max']:.1f}°C")
        print(f"  Humidity: {day['humidity']['avg']:.0f}% (Range: {day['humidity']['min']}% - {day['humidity']['max']}%)")
        print(f"  Cloud Cover: {day['clouds']['avg']:.0f}%")
        print(f"  Precipitation Chance: {day['precipitation_chance']:.0f}%")
        
        if 'rain' in day and day['rain']['total'] > 0:
            print(f"  Expected Rainfall: {day['rain']['total']:.1f} mm")
            
        # Display some hourly details
        print("\n  Hourly forecast highlights:")
        morning = None
        noon = None
        evening = None
        
        # Find entries close to morning (8-9am), noon (12-1pm), and evening (6-7pm)
        for hour_data in day['hourly']:
            hour = datetime.strptime(hour_data['timestamp'], '%Y-%m-%d %H:%M:%S').hour
            if 8 <= hour <= 9 and not morning:
                morning = hour_data
            elif 12 <= hour <= 13 and not noon:
                noon = hour_data
            elif 18 <= hour <= 19 and not evening:
                evening = hour_data
        
        time_slots = []
        if morning:
            time_slots.append(("Morning", morning))
        if noon:
            time_slots.append(("Noon", noon))
        if evening:
            time_slots.append(("Evening", evening))
        
        for time_name, data in time_slots:
            hour = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S').hour
            print(f"    {time_name} ({hour}:00): {data['temperature']:.1f}°C, {data['description'].capitalize()}, "
                  f"Wind: {data['wind']['speed']} m/s, Precip: {data['pop']:.0f}%")
    
    print(f"\n{'='*60}")


def clear_console():
    """
    Clear the console screen using os module.
    Works on Windows, macOS, and Linux.
    """
    import os
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def clear_console():
    """
    Clear the console screen using os module.
    Works on Windows, macOS, and Linux.
    """
    import os
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def run_weather_advisor():
    """
    Main function to run the Weather Advisor application using standard input instead of PyInputPlus.
    """
    clear_console()
    
    print("\n" + "="*60)
    print("       WELCOME TO THE WEATHER ADVISOR APPLICATION")
    print("="*60)
    print("\nThis application allows you to:")
    print("  1. Check current weather for any location")
    print("  2. View weather forecasts for up to 5 days")
    print("  3. Ask natural language questions about the weather")
    print("  4. View weather data visualizations")
    print("\nAll data is provided in metric units (°C, m/s, mm, etc.)")
    print("\nLet's get started!")
    
    # Initialize with a default location
    location = input("\nEnter a location (city name): ")
    weather_data = get_weather_data(location)
    
    if "error" in weather_data:
        print(f"\nError retrieving weather data: {weather_data['error']}")
        location = input("\nPlease try a different location: ")
        weather_data = get_weather_data(location)
        if "error" in weather_data:
            print(f"\nError retrieving weather data: {weather_data['error']}")
            print("\nExiting application. Please try again later.")
            return
    
    while True:
        clear_console()
        
        print("\n" + "="*60)
        print(f"       WEATHER ADVISOR: {weather_data['location']['name'].upper()}, {weather_data['location']['country']}")
        print("="*60)
        
        menu_options = [
            'View Current Weather',
            'View Forecast',
            'Ask a Weather Question',
            'View Temperature Visualization',
            'View Precipitation Visualization',
            'View Wind Visualization',
            'Change Location',
            'Exit'
        ]
        
        print("\nWhat would you like to do?")
        for i, option in enumerate(menu_options, 1):
            print(f"{i}. {option}")
        
        # Input validation for menu choice
        while True:
            try:
                choice = int(input("\nEnter your choice (1-8): "))
                if 1 <= choice <= 8:
                    break
                else:
                    print("Please enter a number between 1 and 8.")
            except ValueError:
                print("Please enter a valid number.")
        
        if choice == 1:  # View Current Weather
            clear_console()
            display_current_weather(weather_data)
            input("Press Enter to continue...")
            
        elif choice == 2:  # View Forecast
            clear_console()
            display_forecast(weather_data)
            input("Press Enter to continue...")
            
        elif choice == 3:  # Ask a Weather Question
            clear_console()
            print("\n" + "="*60)
            print("              ASK A WEATHER QUESTION")
            print("="*60)
            print("\nExamples:")
            print("- Will it rain tomorrow in this city?")
            print("- What's the temperature going to be like this weekend?")
            print("- How windy is it right now?")
            print("- What's the forecast for next Tuesday?")
            
            question = input("\nYour weather question: ")
            parsed_question = parse_weather_question(question)
            
            # If no location was found in the question, use the current one
            if not parsed_question["location"]:
                parsed_question["location"] = weather_data["location"]["name"]
            
            # If the location in the question is different from the current one, fetch new data
            if (parsed_question["location"].lower() != weather_data["location"]["name"].lower() and
                parsed_question["location"].lower() != f"{weather_data['location']['name']}, {weather_data['location']['country']}".lower()):
                print(f"\nFetching weather data for {parsed_question['location']}...")
                new_weather_data = get_weather_data(parsed_question["location"])
                if "error" not in new_weather_data:
                    weather_data = new_weather_data
                else:
                    print(f"\nCouldn't find weather data for {parsed_question['location']}. Using current location instead.")
            
            response = generate_weather_response(parsed_question, weather_data)
            print(f"\nResponse: {response}")
            
            input("\nPress Enter to continue...")
            
        elif choice == 4:  # View Temperature Visualization
            clear_console()
            print("\nGenerating temperature visualization...")
            create_temperature_visualisation(weather_data)
            input("\nPress Enter to continue...")
            
        elif choice == 5:  # View Precipitation Visualization
            clear_console()
            print("\nGenerating precipitation visualization...")
            create_precipitation_visualisation(weather_data)
            input("\nPress Enter to continue...")
            
        elif choice == 6:  # View Wind Visualization
            clear_console()
            print("\nGenerating wind visualization...")
            create_wind_visualisation(weather_data)
            input("\nPress Enter to continue...")
            
        elif choice == 7:  # Change Location
            new_location = input("\nEnter a new location (city name): ")
            print(f"\nFetching weather data for {new_location}...")
            new_weather_data = get_weather_data(new_location)
            
            if "error" in new_weather_data:
                print(f"\nError: {new_weather_data['error']}")
                input("Press Enter to continue...")
            else:
                weather_data = new_weather_data
                print(f"\nLocation changed to {weather_data['location']['name']}, {weather_data['location']['country']}")
                input("Press Enter to continue...")
                
        elif choice == 8:  # Exit
            print("\nThank you for using the Weather Advisor Application. Goodbye!")
            break

# Run the application if this script is executed directly
if __name__ == "__main__":
    run_weather_advisor()
