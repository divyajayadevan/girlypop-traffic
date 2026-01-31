import streamlit as st
import cv2
import tempfile
from streamlit_folium import st_folium
from processor import TrafficProcessor
from gis_utils import create_dashboard_map, convert_to_geojson

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeST Traffic GIS", page_icon="🚦", layout="wide")

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Configuration")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.25, 1.0, 0.45)
line_pos = st.sidebar.slider("Detection Line Position", 0.1, 0.9, 0.6)

# --- STATE MANAGEMENT ---
if 'counts' not in st.session_state:
    st.session_state.counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}

# --- MAIN UI ---
st.title("🚦 AI-Driven Traffic Monitoring System")
st.caption("Challenge 1: Automated Detection & Categorization from Static Cameras")

tab_monitor, tab_gis = st.tabs(["📹 Live Analysis", "🗺️ GIS Dashboard"])

# --- TAB 1: MONITORING ---
with tab_monitor:
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        video_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'mov', 'avi'])
    
    with col_stats:
        st.subheader("Real-Time Vehicle Counts")
        # Initialize placeholders
        metric_car = st.empty()
        metric_bike = st.empty()
        metric_heavy = st.empty()
        st.markdown("---")
    
    if video_file:
        processor = TrafficProcessor(confidence=conf_threshold)
        
        # Save uploaded file to temp (Required for OpenCV)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = col_video.empty()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Process Frame
            annotated_frame, updated_counts = processor.process_frame(
                frame, line_pos, st.session_state.counts
            )
            
            # Update State
            st.session_state.counts = updated_counts
            
            # Update Video Feed
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Update Metrics
            metric_car.metric("🚗 Cars", st.session_state.counts["Car"])
            metric_bike.metric("🏍️ Bikes", st.session_state.counts["Bike"])
            metric_heavy.metric("🚛 Heavy Vehicles", st.session_state.counts["Truck"] + st.session_state.counts["Bus"])

        cap.release()

# --- TAB 2: GIS VISUALIZATION ---
with tab_gis:
    st.header("📍 GIS & Analytics Layer")
    
    col_map, col_data = st.columns([2, 1])
    
    with col_map:
        folium_map = create_dashboard_map(st.session_state.counts)
        st_folium(folium_map, width="100%", height=500)
        
    with col_data:
        st.subheader("Export Data")
        st.write("Download processed traffic data for external GIS tools (ArcGIS/QGIS).")
        
        geojson_data = convert_to_geojson(st.session_state.counts)
        st.download_button(
            label="Download GeoJSON",
            data=geojson_data,
            file_name="traffic_data.json",
            mime="application/json"
        )