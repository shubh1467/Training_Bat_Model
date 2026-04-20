from ultralytics import YOLO
import torch

model = YOLO("yolo12-seg.pt")

results = model.train(

    data="/home/ds-khel/Training_Bat_Model/output_dataset/data.yaml",
    task="segment",

    epochs=50,
    patience=40,

    imgsz=640,

    batch=128,                  # ✅ safe for GPU
    workers=2,                # 🔥 reduce CPU load (important)

    device=0 if torch.cuda.is_available() else "cpu",

    optimizer='AdamW',
    lr0=0.0005,
    lrf=0.01,

    box=5.0,
    cls=0.3,
    dfl=1.5,

    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,

    degrees=20,               # 🔥 reduce distortion further
    translate=0.1,
    scale=0.15,               # 🔥 tighter scaling for precision
    shear=0.0,
    perspective=0.0,

    flipud=0.0,
    fliplr=0.5,

    mosaic=0.2,               # 🔥 less distortion → better edges
    mixup=0.0,
    copy_paste=0.0,

    val=True,
    save=True,

    amp=True,
    cache=False,              # 🔥 IMPORTANT → avoids RAM spike

    warmup_epochs=3,

    project="runs_bat_seg",
    name="yolo12_bat_seg_optimized",
    exist_ok=True,
)