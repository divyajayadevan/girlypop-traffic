import folium
from folium.plugins import HeatMap
import pandas as pd
import random

def create_dashboard_map(counts):
    """
    Generates a HeatMap where the intensity is driven by live vehicle counts.
    """
    # 1. Base Location (e.g., MG Road, Kochi)
    center_lat, center_long = 9.9312, 76.2673
    
    # Create the base map
    m = folium.Map(location=[center_lat, center_long], zoom_start=18, tiles="CartoDB dark_matter")
    
    # 2. Calculate Total Traffic Intensity
    # Summing all Incoming/Outgoing keys
    total_vehicles = sum(counts.values())
    
    # 3. GENERATE FAKE HEATMAP DATA
    # We simulate 'total_vehicles' number of GPS signals clustered around the camera.
    # This creates a visual "hotspot" that expands as traffic gets heavier.
    heatmap_data = []
    
    # Add a baseline of 5 'parked cars' so the map isn't empty when count is 0
    simulated_traffic = total_vehicles + 5 
    
    for _ in range(simulated_traffic):
        # random.gauss creates a cluster (Normal Distribution)
        # 0.0006 is roughly a 60-meter spread on the road
        fake_lat = random.gauss(center_lat, 0.0006)
        fake_lon = random.gauss(center_long, 0.0006)
        
        # Format: [Latitude, Longitude, IntensityWeight]
        heatmap_data.append([fake_lat, fake_lon, 1.0])

    # 4. Add the HeatMap Layer
    # Radius: How "blobs" merge. Gradient: Colors from Blue -> Green -> Red
    HeatMap(
        heatmap_data, 
        radius=25, 
        blur=15, 
        min_opacity=0.4,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)

    # 5. Add a Camera Marker for context
    popup_html = f"""
    <div style="font-family: sans-serif;">
        <b>Sensor Node: MG Road</b><br>
        Status: Online 🟢<br>
        Live Count: {total_vehicles}
    </div>
    """
    
    folium.Marker(
        [center_lat, center_long],
        popup=popup_html,
        icon=folium.Icon(color="red", icon="video", prefix="fa")
    ).add_to(m)
    
    return m

def convert_to_geojson(counts):
    # Convert dictionary to a DataFrame and then to JSON string
    df = pd.DataFrame([counts])
    return df.to_json()