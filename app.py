import streamlit as st
import cv2
import tempfile
import torch
from streamlit_folium import st_folium
from processor import TrafficProcessor
from gis_utils import create_dashboard_map, convert_to_geojson

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Autoflow GIS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar completely
)

# --- CUSTOM CSS: FLUENT + MATERIAL + NEW FONTS ---
st.markdown("""
    <style>
        /* 1. IMPORT GOOGLE FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');

        /* 2. MATERIAL DESIGN COLOR PALETTE (Dark Theme) */
        :root {
            --md-sys-color-primary: #D0BCFF;
            --md-sys-color-on-primary: #381E72;
            --md-sys-color-primary-container: #4F378B;
            --md-sys-color-surface: #141218;
            --md-sys-color-surface-variant: #49454F;
            --md-sys-color-outline: #938F99;
        }

        /* 3. APP BACKGROUND */
        .stApp {
            background-color: var(--md-sys-color-surface);
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        }

        /* 4. TYPOGRAPHY SETUP */
        h1, h2, h3 {
            font-family: 'Instrument Serif', serif !important;
            font-weight: 400 !important;
            letter-spacing: 0.05rem;
        }
        
        h1 {
            font-size: 3.5rem !important;
            background: linear-gradient(90deg, #EADDFF, #D0BCFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p, div, label, button, .stMultiSelect, .stSelectbox, .stSlider {
            font-family: 'Atkinson Hyperlegible', sans-serif !important;
            color: #E6E1E5 !important;
        }

        /* 5. FLUENT / MATERIAL GLASS CARDS */
        div[data-testid="stMetric"], .glass-card, .stDataFrame {
            background: rgba(40, 35, 50, 0.4); /* Material Surface Tint */
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            background: rgba(79, 55, 139, 0.2); /* Primary Container Tint */
            border: 1px solid var(--md-sys-color-primary);
            transform: translateY(-2px);
        }

        /* 6. SETTINGS CONTAINER (Top Bar) */
        .settings-container {
            background: rgba(20, 18, 24, 0.6);
            border-radius: 16px;
            padding: 15px 25px;
            margin-bottom: 20px;
            border: 1px solid var(--md-sys-color-outline);
        }

        /* 7. CUSTOM BUTTONS */
        .stButton>button {
            background-color: var(--md-sys-color-primary);
            color: var(--md-sys-color-on-primary) !important;
            border-radius: 100px; /* Material Pill Shape */
            font-weight: 700;
            border: none;
            padding: 0.5rem 1.5rem;
            transition: box-shadow 0.2s;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px var(--md-sys-color-primary);
        }
        
        /* 8. METRIC TEXT COLORS */
        [data-testid="stMetricLabel"] { opacity: 0.8; font-size: 1rem !important; }
        [data-testid="stMetricValue"] { color: var(--md-sys-color-primary) !important; font-size: 2.2rem !important; }

    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'counts' not in st.session_state:
    categories = ["Car", "Bike", "Bus", "Truck"]
    st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in categories}
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()

# --- HEADER & SETTINGS (Moved to Main Page) ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("NeST Traffic GIS")
    st.caption("Real-time AI Traffic Analysis & Geospatial Integration")
with c2:
    if torch.cuda.is_available():
        st.success(f"🚀 GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ CPU Mode")

# --- COLLAPSIBLE SETTINGS BAR ---
with st.expander("⚙️ System Configuration", expanded=True):
    s1, s2, s3 = st.columns([1, 2, 1])
    with s1:
        model_type = st.selectbox("AI Model", ["yolov8n.pt", "yolov8m.pt"], index=0)
    with s2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.35)
    with s3:
        st.write("") # Spacer
        st.write("") 
        stop_button = st.button("🛑 Stop Session", use_container_width=True)

# --- MAIN TABS ---
tab_monitor, tab_gis = st.tabs(["📹 Live Vision", "🗺️ GIS Heatmap"])

with tab_monitor:
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        # Styled Container for Video
        video_file = st.file_uploader("Upload Footage", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")
        st_frame = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats:
        # Incoming Section
        st.markdown("### ⬇️ Incoming")
        c_in_1, c_in_2 = st.columns(2)
        m_in_car = c_in_1.empty()
        m_in_bike = c_in_2.empty()
        m_in_heavy = st.empty()
        
        st.markdown("---")
        
        # Outgoing Section
        st.markdown("### ⬆️ Outgoing")
        c_out_1, c_out_2 = st.columns(2)
        m_out_car = c_out_1.empty()
        m_out_bike = c_out_2.empty()
        m_out_heavy = st.empty()
        
        status_text = st.empty()
    
    if video_file:
        # Reset Stats Logic
        current_file_name = video_file.name
        if 'last_file' not in st.session_state or st.session_state.last_file != current_file_name:
            st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in ["Car", "Bike", "Bus", "Truck"]}
            st.session_state.counted_ids = set()
            st.session_state.last_file = current_file_name

        processor = TrafficProcessor(model_path=model_type, confidence=conf_threshold)
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        status_text.info(f"Processing with {model_type}...")
        
        while cap.isOpened():
            if stop_button:
                status_text.warning("🛑 Stopped by User.")
                break
                
            success, frame = cap.read()
            if not success:
                status_text.success("✅ Analysis Complete.")
                break
            
            # PROCESS
            annotated_frame, updated_counts, updated_ids = processor.process_frame(
                frame, st.session_state.counts, st.session_state.counted_ids
            )
            
            # UPDATE STATE
            st.session_state.counts = updated_counts
            st.session_state.counted_ids = updated_ids
            
            # UI UPDATES
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # INCOMING METRICS
            m_in_car.metric("Cars", st.session_state.counts["Incoming_Car"])
            m_in_bike.metric("Bikes", st.session_state.counts["Incoming_Bike"])
            m_in_heavy.metric("Heavy", st.session_state.counts["Incoming_Truck"] + st.session_state.counts["Incoming_Bus"])
            
            # OUTGOING METRICS
            m_out_car.metric("Cars", st.session_state.counts["Outgoing_Car"])
            m_out_bike.metric("Bikes", st.session_state.counts["Outgoing_Bike"])
            m_out_heavy.metric("Heavy", st.session_state.counts["Outgoing_Truck"] + st.session_state.counts["Outgoing_Bus"])

        cap.release()

with tab_gis:
    st.markdown("### 📍 Live Urban Density Heatmap")
    
    col_map, col_data = st.columns([3, 1])
    
    with col_map:
        folium_map = create_dashboard_map(st.session_state.counts)
        st_folium(folium_map, width="100%", height=600)
        
    with col_data:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Export Data")
        st.caption("Download vector data for ArcGIS/QGIS.")
        
        geojson_data = convert_to_geojson(st.session_state.counts)
        st.download_button(
            label="Download GeoJSON",
            data=geojson_data,
            file_name="traffic_data.json",
            mime="application/json",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)