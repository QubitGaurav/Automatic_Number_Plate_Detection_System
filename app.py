import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import pandas as pd
from pipeline import VehicleIntelligencePipeline

st.set_page_config(
    page_title="KnightSight ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
# Works with Streamlit's light/dark theme instead of overriding it.
# Avoids external font fetches (latency + failure mode on Streamlit Cloud).
# Removes glassmorphism; uses Streamlit's own surface colors via CSS variables.
st.markdown(
    """
    <style>
    /* Layout resets */
    .block-container { padding-top: 1.5rem !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

    /* Top header strip */
    .ks-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .ks-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .ks-sub {
        font-size: 0.85rem;
        opacity: 0.55;
        margin: 2px 0 0;
    }
    .ks-badge {
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 99px;
        border: 1px solid rgba(34,197,94,0.4);
        color: #22c55e;
        white-space: nowrap;
    }

    /* Metric cards */
    .ks-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
    .ks-metric {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 10px;
    }
    .ks-metric-label { font-size: 0.75rem; opacity: 0.5; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .ks-metric-value { font-size: 1.6rem; font-weight: 700; }
    .ks-metric-unit { font-size: 0.85rem; font-weight: 400; opacity: 0.6; margin-left: 2px; }
    .ks-metric-sub { font-size: 0.72rem; opacity: 0.4; margin-top: 2px; }

    /* Plate result rows */
    .ks-plate-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(128,128,128,0.1);
    }
    .ks-plate-row:last-child { border-bottom: none; }
    .ks-plate-num {
        font-family: monospace;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.1em;
    }
    .ks-plate-assoc { font-size: 0.75rem; opacity: 0.45; margin-top: 2px; }
    .ks-badges { display: flex; gap: 6px; }
    .ks-pill {
        font-size: 0.7rem;
        padding: 3px 9px;
        border-radius: 99px;
        font-weight: 500;
    }
    .ks-pill-green { background: rgba(34,197,94,0.12); color: #22c55e; }
    .ks-pill-amber { background: rgba(234,179,8,0.12); color: #ca8a04; }
    .ks-pill-red   { background: rgba(239,68,68,0.12);  color: #ef4444; }

    /* Section header */
    .ks-section {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        opacity: 0.4;
        margin: 1.5rem 0 0.75rem;
    }

    /* Sidebar status dot */
    .ks-status { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; opacity: 0.65; }
    .ks-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline() -> VehicleIntelligencePipeline:
    return VehicleIntelligencePipeline()


def confidence_pill(value: float) -> str:
    pct = f"{value:.1%}"
    if value >= 0.75:
        cls = "ks-pill-green"
    elif value >= 0.50:
        cls = "ks-pill-amber"
    else:
        cls = "ks-pill-red"
    return f'<span class="ks-pill {cls}">{pct}</span>'


def normalize_plate(text: str) -> str:
    return "".join(c for c in text if c.isalnum()).upper()


def render_metrics(inference_s: float, n_vehicles: int, n_plates: int, suffix: str = "image mode"):
    st.markdown(
        f"""
        <div class="ks-metrics">
          <div class="ks-metric">
            <div class="ks-metric-label">Inference time</div>
            <div class="ks-metric-value">{inference_s:.3f}<span class="ks-metric-unit">s</span></div>
            <div class="ks-metric-sub">{suffix}</div>
          </div>
          <div class="ks-metric">
            <div class="ks-metric-label">Vehicles detected</div>
            <div class="ks-metric-value">{n_vehicles}</div>
            <div class="ks-metric-sub">conf ≥ threshold</div>
          </div>
          <div class="ks-metric">
            <div class="ks-metric-label">Plates recognized</div>
            <div class="ks-metric-value">{n_plates}</div>
            <div class="ks-metric-sub">with OCR match</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plate_results(results: list):
    if not results:
        st.info("No plates detected above the confidence threshold.")
        return

    st.markdown('<div class="ks-section">Extracted plates</div>', unsafe_allow_html=True)
    rows_html = ""
    for res in results:
        assoc = "Vehicle matched" if res.get("vehicle_box") else "No vehicle match"
        ocr_pill = confidence_pill(res["ocr_confidence"])
        det_pill = confidence_pill(res["plate_confidence"])
        rows_html += f"""
        <div class="ks-plate-row">
          <div>
            <div class="ks-plate-num">{res['plate_text']}</div>
            <div class="ks-plate-assoc">{assoc}</div>
          </div>
          <div class="ks-badges">
            {ocr_pill}
            {det_pill}
          </div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

    with st.expander("Raw JSON output"):
        st.json(results)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def build_sidebar() -> tuple[str, int, float]:
    with st.sidebar:
        st.markdown("### KnightSight")
        st.markdown('<div class="ks-status"><div class="ks-dot"></div>Pipeline ready</div>', unsafe_allow_html=True)
        st.divider()

        st.markdown("**Media source**")
        media_type = st.radio("Media", ["Image", "Video"], label_visibility="collapsed")

        st.divider()
        st.markdown("**Inference settings**")

        min_confidence = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.35,
            step=0.05,
            help="Minimum detection confidence to include a result.",
        )
        skip_frames = st.slider(
            "Video downsample rate",
            min_value=1,
            max_value=30,
            value=5,
            help="Process 1 frame every N frames. Higher = faster but lower recall.",
        )

        st.divider()
        st.caption("YOLO11 + EasyOCR · Indian ANPR")

    return media_type, skip_frames, min_confidence


# ── Image mode ────────────────────────────────────────────────────────────────

def run_image_mode(pipeline: VehicleIntelligencePipeline, min_confidence: float):
    uploaded = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            """
            <div style="border: 1px dashed rgba(128,128,128,0.25); border-radius: 12px;
                        padding: 3rem; text-align: center; opacity: 0.5; margin-top: 1rem;">
              Drop a JPG, JPEG, or PNG here
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running pipeline…"):
        t0 = time.perf_counter()
        results, vehicles, _ = pipeline.process_image(image_array=img_bgr)
        elapsed = time.perf_counter() - t0

    results = [r for r in results if r["plate_confidence"] >= min_confidence]
    vehicles = [v for v in vehicles if v["confidence"] >= min_confidence]

    annotated = cv2.cvtColor(
        pipeline.annotate_image(img_bgr, results, vehicles), cv2.COLOR_BGR2RGB
    )

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.image(image, caption="Original", use_container_width=True)
    with col_ann:
        st.image(annotated, caption="Annotated (YOLO11)", use_container_width=True)

    render_metrics(elapsed, len(vehicles), len(results))
    render_plate_results(results)

    if results:
        st.markdown('<div class="ks-section">Table view</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": i + 1,
                        "Plate text": r["plate_text"],
                        "OCR confidence": f"{r['ocr_confidence']:.1%}",
                        "Detection confidence": f"{r['plate_confidence']:.1%}",
                        "Vehicle matched": "Yes" if r["vehicle_box"] else "No",
                    }
                    for i, r in enumerate(results)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


# ── Video mode ────────────────────────────────────────────────────────────────

def run_video_mode(pipeline: VehicleIntelligencePipeline, skip_frames: int, min_confidence: float):
    uploaded = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            """
            <div style="border: 1px dashed rgba(128,128,128,0.25); border-radius: 12px;
                        padding: 3rem; text-align: center; opacity: 0.5; margin-top: 1rem;">
              Drop an MP4, AVI, or MOV here
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Write to temp file so OpenCV can seek it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        os.unlink(tmp_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col_feed, col_log = st.columns([3, 2])
    with col_feed:
        frame_ph = st.empty()
        progress = st.progress(0)
        status_txt = st.empty()
    with col_log:
        st.markdown('<div class="ks-section">Live detection log (last 10)</div>', unsafe_allow_html=True)
        log_ph = st.empty()

    history: list[dict] = []
    seen: set[str] = set()
    frame_idx = processed = 0
    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            processed += 1
            results, vehicles, _ = pipeline.process_image(image_array=frame)
            results = [r for r in results if r["plate_confidence"] >= min_confidence]
            vehicles = [v for v in vehicles if v["confidence"] >= min_confidence]

            annotated = cv2.cvtColor(
                pipeline.annotate_image(frame, results, vehicles), cv2.COLOR_BGR2RGB
            )
            frame_ph.image(annotated, caption=f"Frame {frame_idx}", use_container_width=True)

            ts = frame_idx / fps
            for r in results:
                key = normalize_plate(r["plate_text"])
                if key and key not in seen:
                    seen.add(key)
                    history.append(
                        {
                            "Time": f"{ts:.1f}s",
                            "Frame": frame_idx,
                            "Plate": r["plate_text"],
                            "OCR conf": f"{r['ocr_confidence']:.1%}",
                            "Det conf": f"{r['plate_confidence']:.1%}",
                        }
                    )
                    log_ph.dataframe(
                        pd.DataFrame(history).tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )

        pct = min(1.0, frame_idx / max(1, total_frames))
        progress.progress(pct)
        status_txt.caption(f"Frame {frame_idx} / {total_frames}  ·  {pct:.0%}")
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)

    elapsed = time.perf_counter() - t0
    progress.progress(1.0)
    status_txt.caption(f"Done — {processed} frames processed in {elapsed:.1f} s")

    render_metrics(
        elapsed,
        n_vehicles=0,  # Not tracked per-video; set as desired
        n_plates=len(seen),
        suffix=f"{processed} frames · {elapsed / max(1, processed):.3f} s/frame",
    )

    if history:
        st.markdown('<div class="ks-section">Full detection history</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.warning("No plates detected above the confidence threshold.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    st.markdown(
        """
        <div class="ks-header">
          <div>
            <p class="ks-title">🚗 KnightSight ANPR</p>
            <p class="ks-sub">YOLO11 + EasyOCR · Indian license plate recognition</p>
          </div>
          <span class="ks-badge">● Pipeline ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    media_type, skip_frames, min_confidence = build_sidebar()

    with st.spinner("Loading models…"):
        pipeline = load_pipeline()

    if media_type == "Image":
        run_image_mode(pipeline, min_confidence)
    else:
        run_video_mode(pipeline, skip_frames, min_confidence)


if __name__ == "__main__":
    main()