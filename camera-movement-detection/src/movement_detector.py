import cv2
import numpy as np
import logging
from config import THRESHOLDS, MIN_MATCHES, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_orb_stats(frames, names, min_matches):
    orb = cv2.ORB_create(1000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stats = []
    prev_kp, prev_des = orb.detectAndCompute(frames[0], None)
    stats.append({
        'name': names[0],
        'score': 0.0,
        'inliers': len(prev_kp) if prev_kp is not None else 0,
        'outliers': 0
    })
    for i in range(1, len(frames)):
        kp, des = orb.detectAndCompute(frames[i], None)
        name = names[i]
        score = np.inf
        inliers = 0
        outliers = 0
        if des is not None and prev_des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) >= min_matches:
                src = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
                if M is not None and mask is not None:
                    dx, dy = M[0,2], M[1,2]
                    da = np.degrees(np.arctan2(M[1,0], M[0,0]))
                    score = np.hypot(dx, dy) + abs(da)
                    inliers = int(mask.sum())
                    outliers = int(mask.size - mask.sum())
        stats.append({ 'name': name, 'score': score, 'inliers': inliers, 'outliers': outliers })
        prev_kp, prev_des = kp, des
    return stats


def _compute_flow_stats(frames, names, max_corners, quality, min_dist):
    stats = []
    prev = frames[0]
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners,
                                 qualityLevel=quality, minDistance=min_dist)
    stats.append({
        'name': names[0],
        'score': 0.0,
        'inliers': len(p0) if p0 is not None else 0,
        'outliers': 0,
        'movement_type': 'None'
    })
    for i in range(1, len(frames)):
        curr = frames[i]
        name = names[i]
        score = np.inf
        inliers = 0
        outliers = 0
        movement_type = 'None'
        if p0 is not None and len(p0) >= 6:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None)
            if p1 is not None and st is not None:
                good0 = p0[st.squeeze()==1]
                good1 = p1[st.squeeze()==1]
                if len(good0) >= 6:
                    M, mask = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC)
                    if M is not None and mask is not None:
                        dx, dy = M[0,2], M[1,2]
                        da = np.degrees(np.arctan2(M[1,0], M[0,0]))
                        score = np.hypot(dx, dy) + abs(da)
                        inliers = int(mask.sum())
                        outliers = int(mask.size - mask.sum())
                        inlier_ratio = inliers / (inliers + outliers + 1e-6)
                        # Heuristic: if inlier ratio is high and score is high, it's camera movement
                        if score > 5 and inlier_ratio > 0.5:
                            movement_type = 'Camera'
                        elif score > 5 and inlier_ratio <= 0.5:
                            movement_type = 'Object'
                        else:
                            movement_type = 'None'
        stats.append({ 'name': name, 'score': score, 'inliers': inliers, 'outliers': outliers, 'movement_type': movement_type })
        prev = curr
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners,
                                     qualityLevel=quality, minDistance=min_dist)
    return stats


def _compute_homography_stats(frames, names, min_matches):
    orb = cv2.ORB_create(1000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stats = []
    prev_kp, prev_des = orb.detectAndCompute(frames[0], None)
    stats.append({
        'name': names[0],
        'score': 0.0,
        'inliers': len(prev_kp) if prev_kp is not None else 0,
        'outliers': 0
    })
    for i in range(1, len(frames)):
        kp, des = orb.detectAndCompute(frames[i], None)
        name = names[i]
        score = np.inf
        inliers = 0
        outliers = 0
        if des is not None and prev_des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) >= min_matches:
                src = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC)
                if H is not None and mask is not None:
                    dx, dy = H[0,2], H[1,2]
                    score = np.hypot(dx, dy)
                    inliers = int(mask.sum())
                    outliers = int(mask.size - mask.sum())
        stats.append({ 'name': name, 'score': score, 'inliers': inliers, 'outliers': outliers })
        prev_kp, prev_des = kp, des
    return stats


def detect_significant_movement(
    frames, names,
    algorithm: str = None,
    threshold: float = None,
    min_matches: int = None,
    max_corners: int = None,
    quality: float = None,
    min_dist: float = None,
    adaptive_k: float = 2.0
):
    algo = (algorithm or 'ORB')
    mm   = (min_matches or MIN_MATCHES)
    mc   = (max_corners or MAX_CORNERS)
    ql   = (quality     or QUALITY_LEVEL)
    md   = (min_dist    or MIN_DISTANCE)

    if algo == 'ORB':
        stats = _compute_orb_stats(frames, names, mm)
    elif algo == 'OpticalFlow':
        stats = _compute_flow_stats(frames, names, mc, ql, md)
    elif algo == 'Homography':
        stats = _compute_homography_stats(frames, names, mm)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    scores = np.array([s['score'] for s in stats if np.isfinite(s['score'])])
    if threshold is None and len(scores) > 0:
        mu, sigma = scores.mean(), scores.std()
        thresh = mu + adaptive_k * sigma
        logger.info(f'Adaptive threshold: {thresh:.2f} (mean={mu:.2f}, std={sigma:.2f})')
    else:
        thresh = (threshold or THRESHOLDS.get(algo, threshold))

    # Return flagged frames and their movement type
    sig = [s['name'] for s in stats if (not np.isfinite(s['score'])) or s['score'] > thresh]
    sig_types = [(s['name'], s['movement_type']) for s in stats if (not np.isfinite(s['score'])) or s['score'] > thresh]
    logger.info(f"{algo} flagged {len(sig)} frames above threshold {thresh:.2f}")
    return sig, stats, thresh, sig_types