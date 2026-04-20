import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "/home/ds-khel/Training_Bat_Model/runs/segment/runs_bat_seg/yolo26m_bat_seg_optimized/weights/best.pt"
INPUT_VIDEO = "/home/ds-khel/Training_Bat_Model/vid1.mp4"
OUTPUT_VIDEO = INPUT_VIDEO.replace(".mp4", "_16POINT_TRACKED_SIDE.mp4")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (width, height)
)

print("Processing 16-point tracked outline...")

NUM_POINTS = 16
prev_points = None
alpha = 0.8  # smoothing


# -----------------------------
# 🔥 SAMPLE FIXED POINTS FROM CONTOUR
# -----------------------------
def sample_points(contour, num_points=16):
    contour = contour.reshape(-1, 2)

    # cumulative distance
    dists = np.sqrt(((np.diff(contour, axis=0))**2).sum(axis=1))
    dists = np.insert(dists, 0, 0)
    cumulative = np.cumsum(dists)

    if cumulative[-1] == 0:
        return contour.reshape(-1,1,2)

    cumulative /= cumulative[-1]

    target = np.linspace(0, 1, num_points)

    sampled = []
    for t in target:
        idx = np.searchsorted(cumulative, t)

        if idx == 0:
            sampled.append(contour[0])
        elif idx >= len(contour):
            sampled.append(contour[-1])
        else:
            p1 = contour[idx-1]
            p2 = contour[idx]

            ratio = (t - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1] + 1e-6)
            point = (1 - ratio) * p1 + ratio * p2
            sampled.append(point)

    return np.array(sampled, dtype=np.int32).reshape(-1,1,2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    annotated = frame.copy()

    if results.masks is not None:

        for polygon in results.masks.xy:

            contour = polygon.astype(np.int32)

            # -----------------------------
            # Create mask
            # -----------------------------
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            # -----------------------------
            # Get contour
            # -----------------------------
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if len(contours) == 0:
                continue

            best = max(contours, key=cv2.contourArea)

            # -----------------------------
            # 🔥 SAMPLE 16 POINTS
            # -----------------------------
            points = sample_points(best, NUM_POINTS)

            # -----------------------------
            # 🔥 TEMPORAL SMOOTHING
            # -----------------------------
            if prev_points is not None:
                points = (alpha * points + (1 - alpha) * prev_points).astype(np.int32)

            prev_points = points

            # -----------------------------
            # 🔥 DRAW POLYGON
            # -----------------------------
            cv2.polylines(annotated, [points], True, (0,255,0), 3)

            # 🔥 OPTIONAL: draw points
            for p in points:
                x, y = p[0]
                cv2.circle(annotated, (x,y), 3, (0,0,255), -1)

    out.write(annotated)

cap.release()
out.release()

print(f"Done! Output saved at: {OUTPUT_VIDEO}")