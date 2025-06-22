
# ─── Algorithms ───────────────────────────────────────────────────────────────
ALGORITHMS = ["OpticalFlow"]

# ─── Default parameters ───────────────────────────────────────────────────────
DEFAULT_ALGO      = "OpticalFlow"
THRESHOLDS        = {
    "OpticalFlow":   10.0,
}
MIN_MATCHES       = 15          # for ORB
MAX_CORNERS       = 500         # for Optical Flow (Shi–Tomasi)
QUALITY_LEVEL     = 0.01       # for Optical Flow
MIN_DISTANCE      = 7          # for Optical Flow
