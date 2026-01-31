import streamlit as st
import cv2
import tempfile
from streamlit_folium import st_folium
from processor import TrafficProcessor
from gis_utils import create_dashboard_map, convert_to_geojson

st.set_page_config(page_title="NeST Traffic GIS", page_icon="🚦", layout="wide")

st.sidebar.title("⚙️ Configuration")
model_type = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8m.pt"], index=0)
conf_threshold = st.sidebar.slider("Confidence", 0.25, 1.0, 0.35)
stop_button = st.sidebar.button("🛑 Stop")

# Initialize complex state for 2-way counting
if 'counts' not in st.session_state:
    # We now track 8 categories (4 types * 2 directions)
    categories = ["Car", "Bike", "Bus", "Truck"]
    st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in categories}
    
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()

st.title("🚦 AI-Driven Traffic Monitoring System")
st.caption("Double-Directional Detection @ 1080p")

tab_monitor, tab_gis = st.tabs(["📹 Live Analysis", "🗺️ GIS Dashboard"])

with tab_monitor:
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        video_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'mov', 'avi'])
    
    with col_stats:
        # --- SPLIT METRICS UI ---
        st.subheader("⬇️ Incoming (Down)")
        c1, c2, c3 = st.columns(3)
        in_car = c1.empty()
        in_bike = c2.empty()
        in_heavy = c3.empty()
        
        st.markdown("---")
        
        st.subheader("⬆️ Outgoing (Up)")
        c4, c5, c6 = st.columns(3)
        out_car = c4.empty()
        out_bike = c5.empty()
        out_heavy = c6.empty()
        
        status_text = st.empty()
    
    if video_file:
        # Reset Logic
        current_file_name = video_file.name
        if 'last_file' not in st.session_state or st.session_state.last_file != current_file_name:
            st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in ["Car", "Bike", "Bus", "Truck"]}
            st.session_state.counted_ids = set()
            st.session_state.last_file = current_file_name

        processor = TrafficProcessor(model_path=model_type, confidence=conf_threshold)
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = col_video.empty()
        
        while cap.isOpened():
            if stop_button: break
            success, frame = cap.read()
            if not success: break
            
            # Process Frame
            annotated_frame, updated_counts, updated_ids = processor.process_frame(
                frame, st.session_state.counts, st.session_state.counted_ids
            )
            
            st.session_state.counts = updated_counts
            st.session_state.counted_ids = updated_ids
            
            # Display Video
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Display Metrics (Incoming)
            in_car.metric("Cars", st.session_state.counts["Incoming_Car"])
            in_bike.metric("Bikes", st.session_state.counts["Incoming_Bike"])
            in_heavy.metric("Heavy", st.session_state.counts["Incoming_Truck"] + st.session_state.counts["Incoming_Bus"])
            
            # Display Metrics (Outgoing)
            out_car.metric("Cars", st.session_state.counts["Outgoing_Car"])
            out_bike.metric("Bikes", st.session_state.counts["Outgoing_Bike"])
            out_heavy.metric("Heavy", st.session_state.counts["Outgoing_Truck"] + st.session_state.counts["Outgoing_Bus"])

        cap.release()

# --- TAB 2: GIS (Update for combined counts) ---
with tab_gis:
    st.header("📍 GIS Data Export")
    # Helper to sum up total traffic for the map
    total_traffic = sum(st.session_state.counts.values())
    
    # Simple map showing total density
    # (For a refined app, you could add arrows for direction, but a heat dot is safer for now)
    folium_map = create_dashboard_map({"Car": total_traffic, "Bike": 0}) # Hack to reuse existing utils
    st_folium(folium_map, width="100%", height=500)
    
    geojson_data = convert_to_geojson(st.session_state.counts)
    st.download_button("Download Full Report (JSON)", geojson_data, "traffic_report.json", "application/json")