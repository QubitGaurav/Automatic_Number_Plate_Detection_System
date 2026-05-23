import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import tempfile
import os
import time
import pandas as pd
from pipeline import VehicleIntelligencePipeline

# Set page config at the very beginning
st.set_page_config(
    page_title="KnightSight ANPR Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for Outfit font and Glassmorphism dashboard aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient styling */
    .title-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(30,30,40,0.7) 0%, rgba(15,15,25,0.7) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF901E 0%, #00D7FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8C96A6;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* Custom Card */
    .glass-card {
        background: rgba(30, 32, 45, 0.55);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #00D7FF;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Styled metric values */
    .highlight-val {
        font-size: 2rem;
        font-weight: 700;
        color: #FF901E;
    }
    </style>
""", unsafe_allow_html=True)

# Cache pipeline loading to run quickly and save RAM on Streamlit Cloud
@st.cache_resource
def load_pipeline():
    return VehicleIntelligencePipeline()

def main():
    # Render premium header
    st.markdown("""
        <div class="title-container">
            <h1 class="main-title">🚗 KnightSight ANPR Pipeline</h1>
            <p class="subtitle">Next-Gen Vehicle Intelligence & Indian License Plate Extraction (YOLO11 + EasyOCR)</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar parameters
    st.sidebar.markdown("### 🛠️ Configuration")
    media_type = st.sidebar.radio("Select Media Source", ["🖼️ Image", "🎥 Video"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Inference Settings")
    skip_frames = st.sidebar.slider("Video Downsampling Rate", 1, 30, 5, 
                                    help="Process one frame every N frames to optimize run time.")
    min_confidence = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    
    # Load pipeline
    with st.spinner("Initializing YOLO11 and OCR Engines..."):
        pipeline = load_pipeline()

    if media_type == "🖼️ Image":
        st.markdown('<div class="glass-card"><div class="card-title">🖼️ Upload Image</div>'
                    'Upload a JPG, JPEG, or PNG image of a vehicle.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                # Convert file upload to image array
                image = Image.open(uploaded_file).convert('RGB')
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                with st.spinner("Processing Image..."):
                    start_time = time.time()
                    results, vehicles, plates = pipeline.process_image(image_array=img_bgr)
                    elapsed = time.time() - start_time
                
                # Apply custom confidence filtering on results
                filtered_results = [r for r in results if r['plate_confidence'] >= min_confidence]
                filtered_vehicles = [v for v in vehicles if v['confidence'] >= min_confidence]
                
                # Annotate image
                annotated_img = pipeline.annotate_image(img_bgr, filtered_results, filtered_vehicles)
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # Layout results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Input", use_container_width=True)
                with col2:
                    st.image(annotated_rgb, caption="Annotated Result (YOLO11 Detection)", use_container_width=True)

                # Analysis Metrics Cards
                st.markdown("### 📊 Inference Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">⏱️ Inference Time</div>
                            <div class="highlight-val">{elapsed:.3f} s</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">🚗 Vehicles Detected</div>
                            <div class="highlight-val">{len(filtered_vehicles)}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">📇 Plates Recognized</div>
                            <div class="highlight-val">{len(filtered_results)}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Structured Table output
                if filtered_results:
                    st.markdown("### 📇 Extracted License Plate Details")
                    table_data = []
                    for idx, res in enumerate(filtered_results):
                        table_data.append({
                            "Plate #": idx + 1,
                            "Detected Text": res['plate_text'],
                            "OCR Confidence": f"{res['ocr_confidence']:.1%}",
                            "Plate Conf": f"{res['plate_confidence']:.1%}",
                            "Vehicle Association": "Yes" if res['vehicle_box'] else "No"
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No license plates detected at the current confidence threshold.")

                # JSON Raw Output in expander
                with st.expander("🛠️ Raw Pipeline JSON Output"):
                    st.json(filtered_results)

            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif media_type == "🎥 Video":
        st.markdown('<div class="glass-card"><div class="card-title">🎥 Upload Video</div>'
                    'Upload an MP4, AVI, or MOV video stream of traffic.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                # Save uploaded file to temp file to read via OpenCV cap
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_video_path = temp_file.name

                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    st.error("Failed to open the video file.")
                    return

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps):
                    fps = 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                st.markdown("### ⚡ Real-Time Pipeline Execution")
                
                # Image placeholder for real-time video stream visualization
                col_view, col_table = st.columns([3, 2])
                
                with col_view:
                    frame_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                
                with col_table:
                    st.markdown("#### 📝 Detected Plates Log")
                    table_placeholder = st.empty()

                detected_plates_history = []
                unique_plates = set()
                
                frame_idx = 0
                processed_count = 0
                start_time = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Skip frames to optimize CPU processing on Streamlit Cloud
                    if frame_idx % skip_frames == 0:
                        processed_count += 1
                        
                        # Process frame
                        results, vehicles, plates = pipeline.process_image(image_array=frame)
                        
                        # Filter detections
                        filtered_results = [r for r in results if r['plate_confidence'] >= min_confidence]
                        filtered_vehicles = [v for v in vehicles if v['confidence'] >= min_confidence]
                        
                        # Annotate frame
                        annotated_frame = pipeline.annotate_image(frame, filtered_results, filtered_vehicles)
                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame in Streamlit
                        frame_placeholder.image(annotated_rgb, caption="Processed Video Feed", use_container_width=True)

                        # Update detection log
                        timestamp = frame_idx / fps
                        for res in filtered_results:
                            txt = res['plate_text'].strip()
                            if txt:
                                # Normalizing to avoid duplicating slightly messy scans
                                clean_txt = "".join(c for c in txt if c.isalnum()).upper()
                                if clean_txt and clean_txt not in unique_plates:
                                    unique_plates.add(clean_txt)
                                    detected_plates_history.append({
                                        "Time (s)": f"{timestamp:.2f}s",
                                        "Frame": frame_idx,
                                        "Plate Text": txt,
                                        "OCR Confidence": f"{res['ocr_confidence']:.1%}",
                                        "Plate Conf": f"{res['plate_confidence']:.1%}"
                                    })
                                    
                                    # Update Table log real-time
                                    df_log = pd.DataFrame(detected_plates_history).tail(10) # Show last 10 detections
                                    table_placeholder.dataframe(df_log, use_container_width=True, hide_index=True)

                    # Update progress bar
                    progress_pct = min(1.0, frame_idx / max(1, total_frames))
                    progress_bar.progress(progress_pct)
                    progress_text.text(f"Processing frame {frame_idx}/{total_frames} ({progress_pct:.1%})")
                    
                    frame_idx += 1

                cap.release()
                os.unlink(temp_video_path) # Clean up temp file immediately

                elapsed_total = time.time() - start_time
                progress_bar.progress(1.0)
                progress_text.text(f"Completed! Processed {processed_count} frames in {elapsed_total:.2f} seconds.")

                # Final Summary Statistics
                st.markdown("### 🏆 Processing Summary")
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                with sum_col1:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">⏱️ Total Time</div>
                            <div class="highlight-val">{elapsed_total:.2f} s</div>
                        </div>
                    """, unsafe_allow_html=True)
                with sum_col2:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">🎥 Avg Frame Processing Time</div>
                            <div class="highlight-val">{elapsed_total / max(1, processed_count):.3f} s</div>
                        </div>
                    """, unsafe_allow_html=True)
                with sum_col3:
                    st.markdown(f"""
                        <div class="glass-card">
                            <div class="card-title">📇 Unique Plates Extracted</div>
                            <div class="highlight-val">{len(unique_plates)}</div>
                        </div>
                    """, unsafe_allow_html=True)

                if detected_plates_history:
                    st.markdown("### 📝 Full Video Detection History")
                    st.dataframe(pd.DataFrame(detected_plates_history), use_container_width=True, hide_index=True)
                else:
                    st.warning("No license plates detected in the video stream.")

            except Exception as e:
                st.error(f"Error processing video stream: {e}")

if __name__ == '__main__':
    main()
