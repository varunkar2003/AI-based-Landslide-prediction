import requests

def get_dem_slope(lat, lon):
    # Fetch DEM and slope data from OpenTopoData
    dem_url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
    response = requests.get(dem_url)
    
    if response.status_code == 200:
        data = response.json()
        dem = data['results'][0]['elevation']
        # Assuming slope data is not directly available, you would need a separate service or calculate it from DEM data over an area.
        # Here we will just return the DEM as an example.
        slope = "Slope data calculation needed"  # Placeholder for actual slope calculation logic
        return dem, slope
    else:
        return None, None

def get_precipitation(lat, lon):
    # Fetch precipitation data from OpenWeatherMap
    api_key = "your_openweather_api_key"  # Make sure to replace with your API key
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(weather_url)
    
    if response.status_code == 200:
        data = response.json()
        precipitation = data.get('rain', {}).get('1h', '0')  # Precipitation in the last hour in mm
        if precipitation == '0':
            return "No precipitation data available for this location."
        else:
            return precipitation
    else:
        return "Error fetching precipitation data."

def main():
    lat = input("Enter latitude: ")
    lon = input("Enter longitude: ")
    
    try:
        # Get DEM and slope
        dem, slope = get_dem_slope(lat, lon)
        if dem:
            print(f"DEM: {dem} meters")
            print(f"Slope: {slope}")
        else:
            print("DEM and slope data not available.")
        
        # Get rainfall precipitation
        precipitation = get_precipitation(lat, lon)
        if precipitation:
            print(f"Precipitation: {precipitation} mm")
        else:
            print("Precipitation data not available.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
