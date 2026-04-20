import cv2
from ultralytics import YOLO


MODEL_PATH = "/home/ds-khel/Training_Bat_Model/runs/segment/runs_bat_seg/yolo11m_bat_seg_optimized/weights/best.pt"              # your trained model
INPUT_VIDEO = "/home/ds-khel/Training_Bat_Model/vid1.mp4"          # your input video

# Auto output path
OUTPUT_VIDEO = INPUT_VIDEO.replace(".mp4", "_output1.mp4")

model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(INPUT_VIDEO)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print("Processing video...")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO segmentation
    results = model(frame)

    # Plot results (this automatically draws masks + boxes)
    annotated_frame = results[0].plot()

    # Write frame
    out.write(annotated_frame)


cap.release()
out.release()

print(f"Done! Output saved at: {OUTPUT_VIDEO}")