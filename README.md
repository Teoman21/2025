# Camera Movement Detection

A web application for detecting significant global camera movement (shake, pan, tilt) in video sequences. It uses optical flow and RANSAC to distinguish true camera shifts from object motion and highlights frames where the camera itself moves.

---

## Overview

This project implements a camera movement detection component suitable for smart camera systems. Given a video file, it:

1. **Loads** frames from the video file.
2. **Tracks** feature points across consecutive frames using Shi–Tomasi corner detection and Lucas–Kanade optical flow.
3. **Estimates** a global affine transform with RANSAC to measure translation and rotation between frames.
4. **Computes** a movement score for each frame pair by combining translation distance and rotation angle.
5. **Flags** frames where the score exceeds a configurable threshold (static or adaptive).
6. **Classifies** flagged frames as camera movement versus object movement by comparing RANSAC inlier and outlier counts.

A Streamlit front end wraps the algorithm, allowing users to upload a video, adjust parameters, and visualize results.

---

## Features

* Global movement detection using optical flow and RANSAC
* Adaptive thresholding (mean + k·std) for automatic sensitivity
* Camera vs object motion classification via inlier/outlier ratio
* Interactive UI built with Streamlit
* Modular code structure: `config.py`, `prep_data.py`, `movement_detector.py`, `app.py`
* A user-friendly interface for non-technical users that explains what each parameter does when they hover their mouse over the “?” icon.

---

## Requirements

* Python 3.8 or higher
* OpenCV (`opencv-python`)
* NumPy
* Streamlit

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone** the repository:

```bash
git clone https://github.com/Teoman21/2025.git
cd 2025 
cd camera-movement-detection
cd app
streamlit run app.py
```
````

2. **Install** dependencies:
   ```bash
cd 2025
pip install -r requirements.txt
````

---

## Usage

Run the Streamlit application locally:

```bash
streamlit run app/app.py
```

1. In the sidebar, upload a video file (`.mp4`, `.avi`, `.mov`).
2. (Optional) Expand **Advanced Settings** to tune:

   * Movement threshold
   * Number of corners, quality level, minimum distance for feature tracking
   * Frame resizing and maximum frames to process
3. Click **Run Movement Detection**.
4. Review the flagged frame thumbnails and the frame-by-frame statistics table.

---

## Project Structure

```
2025/
├── config.py              # Default algorithm parameters
├── requirements.txt       # Python dependencies
├── src/
│   ├── prep_data.py       # Functions to load video frames
│   └── movement_detector.py # Optical flow + RANSAC detection logic
└── app/
    └── app.py             # Streamlit UI
```

---

## Algorithm Details

### Data Loading

* `load_video()`: reads up to N grayscale frames from a video file using OpenCV.

### Optical Flow and RANSAC

* Detect Shi–Tomasi corners in the first frame.
* Track them in the next frame with Lucas–Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
* Fit an affine transform via `cv2.estimateAffinePartial2D(..., method=RANSAC)` on matched points.

### Movement Scoring

* Extract translation (`dx`, `dy`) and rotation angle (`da`) from the affine matrix.
* Score = √(dx² + dy²) + |da|.

### Thresholding

* **Static**: use a fixed threshold defined in `config.py`.
* 
### Classification

* Inliers ≥ outliers ⇒ camera movement.
* Inliers < outliers ⇒ object movement.

---

## Sample Results

Place your example outputs and supporting materials in the project root as follows:

* **`images/`** – stores all screenshots 
* **`references/`** – stores any prompt files, prompt logs, or other reference materials.



Files in `references/` may not be rendered here but can be reviewed by cloning the repository and inspecting that folder.
