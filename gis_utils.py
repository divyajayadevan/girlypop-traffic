import folium
import pandas as pd
from folium.plugins import MarkerCluster

def create_dashboard_map(counts):
    # Mock Coordinates for the camera location (e.g., Kochi, India)
    center_lat, center_long = 9.9312, 76.2673
    
    m = folium.Map(location=[center_lat, center_long], zoom_start=15)
    
    total_traffic = sum(counts.values())
    
    # Determine Heatmap Color based on Density
    if total_traffic < 20:
        color = "green"
        status = "Low Density"
    elif total_traffic < 50:
        color = "orange"
        status = "Moderate Density"
    else:
        color = "red"
        status = "High Congestion"
    
    popup_html = f"""
    <b>Junction Camera A</b><br>
    Status: {status}<br>
    Total Vehicles: {total_traffic}<br>
    <i>Cars: {counts['Car']} | Bikes: {counts['Bike']}</i>
    """
    
    folium.Marker(
        [center_lat, center_long],
        popup=popup_html,
        icon=folium.Icon(color=color, icon="camera", prefix="fa")
    ).add_to(m)
    
    return m

def convert_to_geojson(counts):
    # Convert dictionary to a DataFrame and then to JSON string
    df = pd.DataFrame([counts])
    return df.to_json()