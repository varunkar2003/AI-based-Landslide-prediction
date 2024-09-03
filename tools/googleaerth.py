import os
import ee
os.environ['EARTHENGINE_API_KEY'] = 'AIzaSyDev4iTrJoRRqmAHb6XnErcHAf3WLK9RM4'


# Authenticate (this might be necessary if you haven't authenticated)
ee.Authenticate()

# Initialize Earth Engine with the API key
ee.Initialize()

# Define the location (Latitude: -13.1631, Longitude: -72.5450)
point = ee.Geometry.Point([-72.5450, -13.1631])

# Load DEM data (SRTM)
dem = ee.Image('USGS/SRTMGL1_003')

# Get the elevation and slope
elevation = dem.reduceRegion(ee.Reducer.mean(), point, scale=30).get('elevation').getInfo()
slope = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), point, scale=30).get('slope').getInfo()

# Load GPM dataset for precipitation
gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V06')

# Filter the data for the point and a specific date range
rainfall = gpm.filterDate('2023-01-01', '2023-12-31').mean().sample(point, 30).first().get('precipitationCal').getInfo()

# Display the results
print(f'Elevation: {elevation} meters')
print(f'Slope: {slope} degrees')
print(f'Average Annual Rainfall: {rainfall} mm/year')

