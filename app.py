import streamlit as st
import cv2
import tempfile
from streamlit_folium import st_folium
from processor import TrafficProcessor
from gis_utils import create_dashboard_map, convert_to_geojson

st.set_page_config(page_title="NeST Traffic GIS", page_icon="🚦", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configuration")
# Default to Nano (n) for speed. Medium (m) is accurate but slow on CPU.
model_type = st.sidebar.selectbox("Model Type", ["yolov8n.pt", "yolov8m.pt"], index=0)
conf_threshold = st.sidebar.slider("AI Confidence", 0.25, 1.0, 0.35)
stop_button = st.sidebar.button("🛑 Stop Processing")

# --- STATE MANAGEMENT ---
if 'counts' not in st.session_state:
    st.session_state.counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()

# --- MAIN UI ---
st.title("🚦 AI-Driven Traffic Monitoring System")
tab_monitor, tab_gis = st.tabs(["📹 Live Analysis", "🗺️ GIS Dashboard"])

with tab_monitor:
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        video_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'mov', 'avi'])
    
    with col_stats:
        st.subheader("Total Unique Vehicles")
        metric_car = st.empty()
        metric_bike = st.empty()
        metric_heavy = st.empty()
        st.markdown("---")
        status_text = st.empty()
    
    if video_file:
        # Reset Logic
        current_file_name = video_file.name
        if 'last_file' not in st.session_state or st.session_state.last_file != current_file_name:
            st.session_state.counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
            st.session_state.counted_ids = set()
            st.session_state.last_file = current_file_name

        processor = TrafficProcessor(model_path=model_type, confidence=conf_threshold)
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = col_video.empty()
        
        status_text.info(f"Processing with {model_type}...")
        
        while cap.isOpened():
            if stop_button:
                status_text.warning("Stopped.")
                break
                
            success, frame = cap.read()
            if not success:
                status_text.success("Finished.")
                break
            
            # --- NEW PROCESS CALL (No Line Position) ---
            annotated_frame, updated_counts, updated_ids = processor.process_frame(
                frame, st.session_state.counts, st.session_state.counted_ids
            )
            
            # Update State
            st.session_state.counts = updated_counts
            st.session_state.counted_ids = updated_ids
            
            # Display
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            metric_car.metric("🚗 Cars", st.session_state.counts["Car"])
            metric_bike.metric("🏍️ Bikes", st.session_state.counts["Bike"])
            metric_heavy.metric("🚛 Heavy Vehicles", st.session_state.counts["Truck"] + st.session_state.counts["Bus"])

        cap.release()

# --- TAB 2: GIS (Unchanged) ---
with tab_gis:
    st.header("📍 GIS & Analytics Layer")
    col_map, col_data = st.columns([2, 1])
    with col_map:
        folium_map = create_dashboard_map(st.session_state.counts)
        st_folium(folium_map, width="100%", height=500)
    with col_data:
        geojson_data = convert_to_geojson(st.session_state.counts)
        st.download_button("Download GeoJSON", geojson_data, "traffic_data.json", "application/json")