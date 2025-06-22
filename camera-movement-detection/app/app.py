# app/app.py

import os
import tempfile

import streamlit as st

# ensure root is on path if needed
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from src.prep_data import load_image_sequence, load_video, capture_from_webcam
from src.movement_detector import detect_significant_movement

st.set_page_config(page_title="Camera Movement Detector", layout="wide")
st.title("📸 Camera Movement Detector (Optical Flow)")

st.markdown("""
Welcome! This tool helps you detect **significant camera movement** (like shaking, tilting, or panning) in a video. It can also distinguish between camera movement and object movement in the scene.

**How to use:**
1. Upload a video.
2. Click **Run Movement Detection**.
3. See which frames have significant movement, and whether it's likely camera or object movement.
""")

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Input Source")
vid = st.sidebar.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov"],
    help="Select a video file to analyze."
)

with st.sidebar.expander("Advanced Settings (optional)"):
    threshold = st.slider(
        "Movement threshold",
        0.0, 100.0, float(config.THRESHOLDS["OpticalFlow"]), step=0.5,
        help="How sensitive the detection is. Lower = more sensitive."
    )
    max_corners = st.slider(
        "Max corners", 100, 2000, config.MAX_CORNERS,
        help="How many points to track for motion."
    )
    quality = st.slider(
        "Quality level", 0.001, 0.1, config.QUALITY_LEVEL, step=0.001,
        help="Minimum quality of tracked points."
    )
    min_dist = st.slider(
        "Min distance", 1, 50, config.MIN_DISTANCE,
        help="Minimum distance between tracked points."
    )
    resize = st.checkbox("Resize frames?", help="Resize images for faster processing.")
    if resize:
        w = st.number_input("Width", 100, 1920, 640)
        h = st.number_input("Height", 100, 1080, 480)
        resize_dim = (w, h)
    else:
        resize_dim = None
    max_frames = st.number_input(
        "Max frames to process", min_value=1, max_value=500, value=100,
        help="Limits how many frames are analyzed."
    )

st.sidebar.markdown("---")
run = st.sidebar.button("Run Movement Detection")

# ─── Run detection ─────────────────────────────────────────────────────────
if run:
    st.info("Processing input and running movement detection...")
    if not vid:
        st.error("Please upload a video file to proceed.")
    else:
        # save uploaded video to a temporary file
        tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmpf.write(vid.getbuffer())
        tmpf.flush()

        # load frames from video
        frames, names = load_video(tmpf.name, resize_dim, max_frames)

        if len(frames) < 2:
            st.error("Video must contain at least 2 frames for movement detection.")
        else:
            # run the detector
            sig, stats, used_thresh, sig_types = detect_significant_movement(
                frames, names,
                algorithm="OpticalFlow",
                threshold=threshold,
                max_corners=max_corners,
                quality=quality,
                min_dist=min_dist
            )
            st.success(
                f"Detected significant movement at {len(sig)} frame(s) "
                f"(threshold: {used_thresh:.2f})"
            )

            # display flagged frames
            if sig:
                st.write("**Flagged frames:**")
                cols = st.columns(4)
                for idx, (frame_name, movement_type) in enumerate(sig_types[:8]):
                    frame_pos = names.index(frame_name)
                    img = frames[frame_pos]
                    label = f"Frame {frame_name}"
                    cols[idx % 4].image(
                        img,
                        caption=label,
                        use_container_width=True
                    )
            else:
                st.info("No significant camera movement detected.")

            # show detailed stats
            st.markdown("---")
            st.write("### All Frame Movement Scores")
            st.dataframe({
                "Frame":   [s["name"]     for s in stats],
                "Score":   [s["score"]    for s in stats],
                "Inliers": [s["inliers"]  for s in stats],
                "Outliers":[s["outliers"] for s in stats],
                "Type":    [t for _, t in sig_types],
            }, use_container_width=True)
