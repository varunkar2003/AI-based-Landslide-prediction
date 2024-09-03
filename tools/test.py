import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import json

def get_location_name(lat, lon):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.reverse((lat, lon), language='en')
    return location.address

def scrape_dem(lat, lon):
    # Use the correct endpoint or manually check for the correct API link
    # Assuming we have found the correct API endpoint
    url = f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=meters&output=json"
    response = requests.get(url, allow_redirects=True)  # Allow redirects if necessary
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return None
    
    print(f"Response Text: {response.text}")  # Debugging print
    
    try:
        data = response.json()
        dem_value = data['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
        return dem_value
    except ValueError as e:
        print(f"JSON decode error: {e}")
        return None


def scrape_slope(lat, lon):
    # Example slope data calculation (replace with actual source if available)
    dem = scrape_dem(lat, lon)
    # Slope calculation from DEM is complex and typically requires GIS software.
    # Placeholder: Slope is often calculated using surrounding DEM points.
    slope_value = dem * 0.1  # This is a placeholder calculation
    return slope_value

def scrape_ndvi(lat, lon):
    # Example NDVI data using Google Earth Engine (GEE) (placeholder)
    url = f"https://api.example.com/ndvi?lat={lat}&lon={lon}"  # Replace with actual GEE endpoint
    response = requests.get(url)
    data = response.json()
    ndvi_value = data['ndvi']  # Adjust this based on actual response structure
    return ndvi_value

def get_dem_slope_ndvi(lat, lon):
    try:
        location_name = get_location_name(lat, lon)
        dem = scrape_dem(lat, lon)
        slope = scrape_slope(lat, lon)
        ndvi = scrape_ndvi(lat, lon)

        print(f"Location: {location_name}")
        print(f"DEM: {dem} meters")
        print(f"Slope: {slope} degrees (approx.)")
        print(f"NDVI: {ndvi}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
latitude = 34.0522  # Replace with actual latitude
longitude = -118.2437  # Replace with actual longitude

get_dem_slope_ndvi(latitude, longitude)
